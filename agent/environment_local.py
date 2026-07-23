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
                 alpha: float = 10.0, beta: float = 2.0,
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
        self.n_candidates = len(self.candidates)                     # <-- nouveau
        self.templates = templates
        self.n_templates = n_templates
        self.storage_budget = storage_budget
        self.alpha = alpha
        self.beta = beta
        self.agent_type = agent_type.lower()
        
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
        self._spaces_used = 0.0                     # total space used in bytes
        self._candidate_sizes = {}                  # cache for index sizes

        self.db_replica = Replica(self.replica_id, self.hostname, self.port, self.db_name, self.user, self.password)
        self.initial_costs = [0 for _ in range(self.n_templates)]

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
        
        # try:
        #     if self.candidates:
        #         tables_to_clean = list(set([c[0] for c in self.candidates if c and len(c) > 0]))
        #         if tables_to_clean:
        #             self.db_replica.drop_all_indexes(tables_to_clean, mode='cost')
        # except Exception as db_err:
        #     print(f"[Worker Environment {self.replica_id} Warning] Impossible de reset les index : {db_err}")
        
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
        return self._get_obs(), {'agent_mode': self.agent_type}
        
    def step(self, action: int, queries=None):
        """
        Execute one local indexing action (add/drop) given a specific sub-workload slice.
        This version implements a hard storage budget constraint similar to the original qDINA:
        - Before adding an index, we check if the required space fits in the remaining budget.
        - If it does not fit, the action is rejected (state unchanged) and the episode terminates.
        - The reward penalizes both performance loss and storage usage.
        - Index sizes are computed using HypoPG and cached for efficiency.
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
            reward = 0.0
            terminated = False
            truncated = False
            return self._get_obs(), reward, terminated, truncated, {
                'costs': current_costs,
                'total_cost': current_total,
                'storage': self._spaces_used,
                'agent_mode': self.agent_type
            }

        self.initial_costs = self._estimate_workload_costs(queries)
        initial_total = sum(self.initial_costs)

        if self._current_indexes[action] == 0:
            candidate = self.candidates[action]
            required_space = self._get_candidate_size(candidate)
            if self._spaces_used + required_space > self.storage_budget:
                reward = -10.0
                terminated = False
                truncated = False
                return self._get_obs(), reward, terminated, truncated, {
                    'costs': self.initial_costs,
                    'total_cost': initial_total,
                    'storage': self._spaces_used,
                    'agent_mode': self.agent_type
                }
            else:
                self._current_indexes[action] = 1
                self._spaces_used += required_space
        else:
            candidate = self.candidates[action]
            size = self._get_candidate_size(candidate)
            self._current_indexes[action] = 0
            self._spaces_used -= size

        current_costs = self._estimate_workload_costs(queries)
        current_total = sum(current_costs)
        self.last_costs = current_costs[:]

        used_storage = self._spaces_used

        initial_total = sum(self.initial_costs)
        current_total = sum(current_costs)

        if initial_total > 0 and current_total > 0:
            reward_t = (initial_total - current_total) / initial_total 
        else:
            reward_t = 0.0

        reward_s = max(0.0, (self.storage_budget - used_storage) / self.storage_budget)
        # reward = (self.alpha * reward_t) + (self.beta * reward_s)

        cost_saving = initial_total - current_total
        storage_penalty = self.beta * (used_storage / self.storage_budget) ** 2
        reward = cost_saving - storage_penalty

        terminated = False
        truncated = False
        return self._get_obs(), reward, terminated, truncated, {
            'costs': current_costs,
            'total_cost': sum(current_costs),
            'storage': used_storage,
            'agent_mode': self.agent_type
        }
    
    def _get_obs(self):
        costs_norm = np.log10(np.array(self.last_costs, dtype=np.float32) + 1.0)
        return np.concatenate([
            self._current_workload_state.astype(np.float32),
            self._current_indexes.astype(np.float32),
            costs_norm
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