# File: agent/environment_local.py
import gymnasium as gym
import numpy as np
from multiprocessing import Queue, Process
from agent.database import Replica
from agent.cost_estimator import CostEstimator

class LocalIndexingEnv(gym.Env):
    def __init__(self, replica_id: int, hostname: str, port: int, user: str, password: str,
                 db_name:str, candidates: list, templates: list[int], 
                 n_templates: int, storage_budget: float,
                 alpha: float = 10.0, beta: float = 0.5,
                 agent_type: str = 'classical'):
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
        self.templates = templates
        self.n_templates = n_templates
        self.storage_budget = storage_budget
        self.alpha = alpha
        self.beta = beta
        self.agent_type = agent_type.lower()
        
        self.n_actions = len(self.candidates)
        self.action_space = gym.spaces.Discrete(self.n_actions)
        
        self.observation_space = gym.spaces.Box(
            low=0, high=1000,
            shape=(self.n_templates + self.n_actions,),
            dtype=np.float32
        )

        self._current_indexes = np.zeros(self.n_actions)
        self._current_workload_state = np.zeros(self.n_templates, dtype=np.int32)
        
        # Attributes for real storage budget management
        self._spaces_used = 0.0                     # total space used in bytes
        self._candidate_sizes = {}                  # cache for index sizes

        self.db_replica = Replica(self.replica_id, self.hostname, self.port, self.db_name, self.user, self.password)
        self.initial_costs = [0 for _ in range(self.n_templates)]

        self.index_gains = {} 
        
    def _get_candidate_size(self, candidate) -> int:
        """
        Compute the real size (in bytes) of a candidate index using HypoPG.
        The result is cached to avoid repeated database calls.
        This follows the approach used in the original DINA/qDINA environment.
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
        if not queries:
            return [0 for _ in range(self.n_templates)]
        
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
        
        try:
            if self.candidates:
                tables_to_clean = list(set([c[0] for c in self.candidates if c and len(c) > 0]))
                if tables_to_clean:
                    self.db_replica.drop_all_indexes(tables_to_clean, mode='cost')
        except Exception as db_err:
            print(f"[Worker Environment {self.replica_id} Warning] Impossible de reset les index : {db_err}")
        
        self._current_indexes = np.zeros(self.n_actions)
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
        
        return self._get_obs(), {'agent_mode': self.agent_type}
        
    def step(self, action: int, queries=None):
        """
        Execute one local indexing action (add/drop) given a specific sub-workload slice.
        This version removes the hard storage budget limit during training, but tracks
        the gain (cost reduction) of each index to allow a final knapsack selection.
        The reward penalizes both performance loss and storage usage.
        """
        if queries is None:
            queries = []

        # Update workload state (template frequencies)
        self._current_workload_state = np.zeros(self.n_templates, dtype=np.int32)
        for q_idx in range(len(queries)):
            if q_idx < len(self.templates):
                t_id = self.templates[q_idx]
                if 0 <= t_id < self.n_templates:
                    self._current_workload_state[t_id] += 1

        # Clear any virtual indexes from previous steps
        tables_to_clean = list(set([c[0] for c in self.candidates if c and len(c) > 0]))
        if tables_to_clean:
            self.db_replica.drop_all_indexes(tables_to_clean, mode='cost')

        # Cost before applying the action 
        initial_costs = self._estimate_workload_costs(queries)
        initial_total = sum(initial_costs)

        # Apply the action (add or drop the selected index) 
        if self._current_indexes[action] == 0:
            # Add index
            candidate = self.candidates[action]
            required_space = self._get_candidate_size(candidate)
            self._current_indexes[action] = 1
            self._spaces_used += required_space
            # We'll compute the gain after estimating costs with the new index
        else:
            # Drop index
            candidate = self.candidates[action]
            size = self._get_candidate_size(candidate)
            self._current_indexes[action] = 0
            self._spaces_used -= size

        # Cost after applying the action
        if tables_to_clean:
            self.db_replica.drop_all_indexes(tables_to_clean, mode='cost')
        current_costs = self._estimate_workload_costs(queries)
        current_total = sum(current_costs)

        knapsack_indexes = self.get_knapsack_selection()  # liste de (table, colonnes)
        costs_knapsack = self._estimate_cost_with_indexes(queries, knapsack_indexes)

        # Compute the gain (cost reduction) attributable to this action
        gain = initial_total - current_total  # Positive if costs decreased

        # Store the gain if it's positive and we are adding an index;
        # if we are dropping, remove the entry (if any) because the index is no longer active.
        if gain > 0 and self._current_indexes[action] == 1:
            self.index_gains[action] = gain
        else:
            # If the gain is non-positive or we dropped, discard the stored gain.
            self.index_gains.pop(action, None)

        used_storage = self._spaces_used

        # Reward calculation
        # Performance gain (relative improvement)
        if initial_total > 0 and current_total > 0:
            reward_t = (initial_total - current_total) / initial_total
        else:
            reward_t = 0.0

        # Space penalty (proportional to used space)
        space_penalty = self.beta * (self._spaces_used / self.storage_budget)

        # Combined reward: performance gain minus space penalty
        reward = self.alpha * reward_t - space_penalty

        terminated = False
        truncated = False

        return self._get_obs(), reward, terminated, truncated, {
            'costs': current_costs,
            'costs_knapsack': costs_knapsack,
            'total_cost': current_total,
            'storage': used_storage,
            'agent_mode': self.agent_type
        }
    
    def _get_obs(self):
        return np.concatenate([
            self._current_workload_state.astype(np.float32),
            self._current_indexes.astype(np.float32)
        ])
    
    def get_active_index_names(self):
        """
        Returns the names of the currently active indexes based on the internal state.
        """
        active_indexes = []
        for idx_pos, val in enumerate(self._current_indexes):
            if val == 1:
                table, columns = self.candidates[idx_pos]
                index_name = f"{table}_{'_'.join(columns)}"
                active_indexes.append(index_name)
        return active_indexes

    def get_knapsack_selection(self):
        from common.knapsack import select_indexes_knapsack

        items = []
        for action, gain in self.index_gains.items():
            if gain > 0:
                size = self._get_candidate_size(self.candidates[action])
                items.append((gain, size, action))
        
        if not items:
            return []
        
        selected_items = select_indexes_knapsack(
            [(gain, size) for gain, size, _ in items],
            self.storage_budget
        )

        selected_actions = []
        for gain, size in selected_items:

            for g, s, act in items:
                if g == gain and s == size:
                    selected_actions.append(act)
                    break
        

        selected_indexes = []
        for act in selected_actions:
            table, columns = self.candidates[act]
            selected_indexes.append((table, columns))
        return selected_indexes


    def _estimate_cost_with_indexes(self, queries, indexes):
        """Estimate costs with a given list of indexes (no modification of internal state)."""
        if not queries:
            return [0.0] * self.n_templates

        local_queue = Queue()
        conn_string = f"host={self.hostname} port={self.port} dbname={self.db_name} user={self.user} password={self.password}"
        estimator = CostEstimator(self.n_templates, conn_string, local_queue)
        p = Process(target=estimator.run, args=(queries, self.templates, indexes))
        p.start()
        costs = local_queue.get(timeout=120)
        p.join()
        return costs