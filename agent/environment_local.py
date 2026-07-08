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
                 alpha: float = 1.0, beta: float = 0.5,
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
        
        self.observation_space = gym.spaces.Box(low=0, high=1000, shape=(self.n_templates,), dtype=np.int32)
        
        self._current_indexes = np.zeros(self.n_actions)
        self._current_workload_state = np.zeros(self.n_templates, dtype=np.int32)
        
        self.db_replica = Replica(self.replica_id, self.hostname, self.port, self.db_name, self.user, self.password)
        self.initial_costs = [0 for _ in range(self.n_templates)]
        
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
        
        return self._current_workload_state, {'agent_mode': self.agent_type}
        
    def step(self, action: int, queries=None):
        """
        Execute one local indexing action (add/drop) given a specific sub-workload slice.
        """
        if queries is None:
            queries = []
            
        self._current_workload_state = np.zeros(self.n_templates, dtype=np.int32)
        for q_idx in range(len(queries)):
            if q_idx < len(self.templates):
                t_id = self.templates[q_idx]
                if 0 <= t_id < self.n_templates:
                    self._current_workload_state[t_id] += 1
                    
        tables_to_drop = getattr(self, 'tables', [])
        self.db_replica.drop_all_indexes(tables_to_drop, 'cost')
        self.initial_costs = self._estimate_workload_costs(queries)

        tables_to_clean = list(set([c[0] for c in self.candidates if c and len(c) > 0]))
        
        if tables_to_clean:
            self.db_replica.drop_all_indexes(tables_to_clean, mode='cost')
            
        self.initial_costs = self._estimate_workload_costs(queries)

        if self._current_indexes[action] == 0:
            self._current_indexes[action] = 1
        else:
            self._current_indexes[action] = 0
            
        if tables_to_clean:
            self.db_replica.drop_all_indexes(tables_to_clean, mode='cost')
            
        current_costs = self._estimate_workload_costs(queries)
        
        used_storage = sum([1.5 for i in range(self.n_actions) if self._current_indexes[i] == 1])
        
        perf_gain = sum(self.initial_costs) - sum(current_costs)
        reward_t = max(0.0, perf_gain)
        reward_s = max(0.0, self.storage_budget - used_storage)
        
        reward = (self.alpha * reward_t) + (self.beta * reward_s)
        
        terminated = False
        if used_storage > self.storage_budget:
            reward = -100.0
            terminated = True
            
        truncated = False
        return self._current_workload_state, reward, terminated, truncated, {
            'costs': current_costs,
            'total_cost': sum(current_costs),
            'storage': used_storage,
            'agent_mode': self.agent_type
        }
    
    def _get_obs(self):
        return self._current_workload_state.copy()