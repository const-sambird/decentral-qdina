# File: agent/environment_local.py
import gymnasium as gym
import numpy as np
from multiprocessing import Queue, Process
from agent.database import Replica
from agent.cost_estimator import CostEstimator

class LocalIndexingEnv(gym.Env):
    def __init__(self, replica_id: int, hostname: str, port: int, user: str, password: str,
                 db_name: str, candidates: list, templates: list[int],
                 n_templates: int, storage_budget: float,
                 alpha: float = 10.0, beta: float = 0.5,
                 agent_type: str = 'classical',
                 max_stagnation_steps: int = 10):
        '''
        Local Environment for a single database replica managing its own indexes.
        Follows the decentralized qDINA architecture where the state represents the incoming sub-workload.
        Supports both Classical (DQN) and Quantum (QNN) execution modes.
        '''
        super(LocalIndexingEnv, self).__init__()
        self.replica_id = replica_id
        self.hostname = hostname
        self.port = port
        self.user = user
        self.password = password
        self.db_name = db_name
        self.candidates = candidates
        self.n_candidates = len(self.candidates)
        self.templates = templates
        self.n_templates = n_templates
        self.storage_budget = storage_budget
        self.alpha = alpha
        self.beta = beta
        self.agent_type = agent_type.lower()
        self.max_stagnation_steps = max_stagnation_steps

        self.n_actions = self.n_candidates + 1
        self.action_space = gym.spaces.Discrete(self.n_actions)

        self.observation_space = gym.spaces.Box(
            low=0, high=1000,
            shape=(n_templates + self.n_candidates + n_templates,),
            dtype=np.float32
        )

        self._current_indexes = np.zeros(self.n_candidates)
        self.last_costs = [0.0] * n_templates
        self._current_workload_state = np.zeros(self.n_templates, dtype=np.int32)

        # Attributes for real storage budget management
        self._spaces_used = 0.0
        self._candidate_sizes = {}

        self.db_replica = Replica(self.replica_id, self.hostname, self.port, self.db_name, self.user, self.password)
        self.initial_costs = [0 for _ in range(self.n_templates)]

        self.penalty_toggle = 1e-6
        self.bonus_noop = 1e-3

        # Stagnation tracking
        self.best_cost_so_far = None
        self.stagnation_counter = 0

        # Best configuration memory
        self.best_indexes = None
        self.best_cost = float('inf')

    def _get_candidate_size(self, candidate) -> int:
        """
        Compute the real size (in bytes) of a candidate index using HypoPG.
        The result is cached to avoid repeated database calls.
        """
        if candidate in self._candidate_sizes:
            return self._candidate_sizes[candidate]

        table = candidate[0]
        columns = candidate[1]
        creation_string = f'CREATE INDEX candidate_index ON {table} ({", ".join(columns)})'

        try:
            conn = self.db_replica.connection()
            with conn.cursor() as cur:
                cur.execute('SELECT indexrelid FROM hypopg_create_index($$%s$$);' % creation_string)
                virtual_oid = cur.fetchone()[0]
                try:
                    cur.execute('SELECT hypopg_relation_size(%s);', (virtual_oid,))
                    size = cur.fetchone()[0]
                except Exception:
                    print(f"[Worker {self.replica_id}] Warning: Unable to get size for candidate {candidate}. Using default size.")
                    size = 5_000_000
                cur.execute('SELECT hypopg_drop_index(%s);' % virtual_oid)
                conn.commit()
                self._candidate_sizes[candidate] = size
                return size
        except Exception as e:
            print(f"[Worker {self.replica_id}] Error getting candidate size for {candidate}: {e}")
            default_size = 5_000_000
            self._candidate_sizes[candidate] = default_size
            return default_size

    def _estimate_workload_costs(self, queries):
        tables_to_clean = list(set([c[0] for c in self.candidates if c and len(c) > 0]))
        if tables_to_clean:
            self.db_replica.drop_all_indexes(tables_to_clean, mode='cost')

        local_queue = Queue()
        active_indexes = []
        for idx_pos, val in enumerate(self._current_indexes):
            if val == 1:
                active_indexes.append(self.candidates[idx_pos])

        conn_string = f"host={self.hostname} port={self.port} dbname={self.db_name} user={self.user} password={self.password}"
        estimator = CostEstimator(self.n_templates, conn_string, local_queue)
        p = Process(target=estimator.run, args=(queries, self.templates, active_indexes))

        try:
            p.start()
            costs = local_queue.get(timeout=120)
            p.join()
            return costs
        except Exception as e:
            print(f"[Worker Indexing Env {self.replica_id} Warning] Échec calcul coûts (Port {self.port}) : {e}")
            if p.is_alive():
                p.terminate()
            return [100000.0] * self.n_templates

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._current_indexes = np.zeros(self.n_candidates)
        self._spaces_used = 0.0

        incoming_queries = []
        if options and 'queries' in options:
            incoming_queries = options['queries']

        self._current_workload_state = np.zeros(self.n_templates, dtype=np.int32)
        for q_idx in range(len(incoming_queries)):
            if q_idx < len(self.templates):
                t_id = self.templates[q_idx]
                if 0 <= t_id < self.n_templates:
                    self._current_workload_state[t_id] += 1

        self.initial_costs = self._estimate_workload_costs(incoming_queries)
        self.last_costs = self.initial_costs[:]

        # Reset stagnation tracking
        self.stagnation_counter = 0
        current_total = sum(self.initial_costs)
        if self.best_cost_so_far is None:
            self.best_cost_so_far = current_total if current_total > 0 else None

        # Reset best configuration memory
        self.best_indexes = None
        self.best_cost = float('inf')

        return self._get_obs(), {'agent_mode': self.agent_type}

    def step(self, action: int, queries=None):
        """
        Execute one local indexing action (add/drop) given a specific sub-workload slice.
        If the cost does not improve significantly for `max_stagnation_steps`, all indexes are cleared.
        Best configurations are memorized and can be restored if stagnation occurs.
        """
        if queries is None:
            queries = []

        self._current_workload_state = np.zeros(self.n_templates, dtype=np.int32)
        for q_idx in range(len(queries)):
            if q_idx < len(self.templates):
                t_id = self.templates[q_idx]
                if 0 <= t_id < self.n_templates:
                    self._current_workload_state[t_id] += 1

        no_op_action = self.n_actions - 1

        if action == no_op_action:
            current_costs = self.last_costs if hasattr(self, 'last_costs') else self.initial_costs
            current_total = sum(current_costs)
            # Reward for stability: bonus if current configuration is good
            if self.best_cost_so_far is not None and current_total <= 1.5 * self.best_cost_so_far:
                reward = 0.5
            else:
                reward = -0.1
            return self._get_obs(), reward, False, False, {
                'costs': current_costs,
                'total_cost': current_total,
                'storage': self._spaces_used,
                'agent_mode': self.agent_type
            }

        # Save old state to know if we are adding or dropping
        old_indexes = self._current_indexes.copy()
        old_storage = self._spaces_used

        # Estimate initial costs (without the modification)
        self.initial_costs = self._estimate_workload_costs(queries)
        initial_total = sum(self.initial_costs)

        # Apply the action (add or drop)
        if self._current_indexes[action] == 0:
            # Adding an index
            candidate = self.candidates[action]
            required_space = self._get_candidate_size(candidate)
            if self._spaces_used + required_space > self.storage_budget:
                reward = -10.0
                return self._get_obs(), reward, False, False, {
                    'costs': self.initial_costs,
                    'total_cost': initial_total,
                    'storage': self._spaces_used,
                    'agent_mode': self.agent_type
                }
            else:
                self._current_indexes[action] = 1
                self._spaces_used += required_space
        else:
            # Dropping an index
            candidate = self.candidates[action]
            size = self._get_candidate_size(candidate)
            self._current_indexes[action] = 0
            self._spaces_used -= size

        # Estimate costs after modification
        current_costs = self._estimate_workload_costs(queries)
        current_total = sum(current_costs)
        self.last_costs = current_costs[:]

        used_storage = self._spaces_used

        # Improved reward with non-linear scaling for huge costs
        storage_penalty = self.beta * (used_storage / self.storage_budget) ** 2
        active_count = np.sum(self._current_indexes)
        active_index_penalty = 0.05 * active_count
        toggle_penalty = 0.02

        if initial_total > 0:
            normalized_cost_saving = (initial_total - current_total) / initial_total
            normalized_cost_impact = (current_total - initial_total) / initial_total
        else:
            normalized_cost_saving = 0.0
            normalized_cost_impact = 0.0

        # Non-linear transformation to heavily penalize cost increases and reward decreases
        if old_indexes[action] == 1:
            # Dropping an index
            if normalized_cost_impact > 0:
                # Quadratic penalty: (1 + impact)^2 - 1
                penalty_cost = (1.0 + normalized_cost_impact) ** 2 - 1.0
            else:
                # If cost decreased (rare), small reward
                penalty_cost = -0.1 * normalized_cost_saving
            bonus_drop = 1.0
            reward = -penalty_cost - storage_penalty - toggle_penalty - active_index_penalty + bonus_drop
        else:
            # Adding an index
            if normalized_cost_saving > 0.01:
                # Reward using log to keep it bounded
                reward = np.log1p(normalized_cost_saving) - storage_penalty - toggle_penalty - active_index_penalty
            else:
                if normalized_cost_impact > 0:
                    penalty_cost = (1.0 + normalized_cost_impact) ** 2 - 1.0
                else:
                    penalty_cost = 0.0
                reward = -penalty_cost - storage_penalty - toggle_penalty - active_index_penalty

        # Best configuration memory
        if current_total < self.best_cost:
            self.best_cost = current_total
            self.best_indexes = self._current_indexes.copy()

        # Stagnation reset logic
        # Only consider improvements greater than 2% of best cost
        improvement_threshold = 0.02
        if self.best_cost_so_far is not None and self.best_cost_so_far > 0:
            improvement_ratio = (self.best_cost_so_far - current_total) / self.best_cost_so_far
            if improvement_ratio > improvement_threshold:
                self.best_cost_so_far = current_total
                self.stagnation_counter = 0
            else:
                self.stagnation_counter += 1
        else:
            if self.best_cost_so_far is None:
                self.best_cost_so_far = current_total

        # Reset if we have stagnated for max_stagnation_steps
        if (self.best_cost_so_far is not None and self.best_cost_so_far > 0 and
            self.stagnation_counter >= self.max_stagnation_steps):
            # If we have a saved best configuration and current cost is much worse, restore it
            if (self.best_indexes is not None and
                current_total > 2.0 * self.best_cost and
                self.best_cost < current_total):
                print(f"[Worker {self.replica_id}] Restoring best configuration with cost {self.best_cost:.2f} (current: {current_total:.2f})")
                self._current_indexes = self.best_indexes.copy()
                self._spaces_used = self._compute_storage_from_indexes()
                # We don't reset best_cost_so_far because we want to keep the best seen
                self.stagnation_counter = 0
            else:
                print(f"[Worker {self.replica_id}] Stagnation detected (no significant improvement for {self.max_stagnation_steps} steps), clearing all indexes.")
                self._current_indexes[:] = 0
                self._spaces_used = 0.0
                self.stagnation_counter = 0
                self.best_cost_so_far = current_total

        terminated = False
        truncated = False
        return self._get_obs(), reward, terminated, truncated, {
            'costs': current_costs,
            'total_cost': current_total,
            'storage': used_storage,
            'agent_mode': self.agent_type
        }

    def _compute_storage_from_indexes(self):
        """Compute the storage used by the current set of active indexes."""
        total = 0.0
        for i, val in enumerate(self._current_indexes):
            if val == 1:
                candidate = self.candidates[i]
                # Use cached size or compute it
                size = self._candidate_sizes.get(candidate, 5_000_000)
                total += size
        return total

    def _get_obs(self):
        costs_norm = np.log10(np.array(self.last_costs, dtype=np.float32) + 1.0)
        return np.concatenate([
            self._current_workload_state.astype(np.float32),
            self._current_indexes.astype(np.float32),
            costs_norm
        ])

    def get_active_index_names(self):
        active_indexes = []
        for idx_pos, val in enumerate(self._current_indexes):
            if val == 1:
                table, columns = self.candidates[idx_pos]
                index_name = f"{table}_{'_'.join(columns)}"
                active_indexes.append(index_name)
        return active_indexes