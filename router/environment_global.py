import gymnasium as gym
import numpy as np

class GlobalRoutingEnv(gym.Env):
    def __init__(self, n_templates: int = 22, n_replicas: int = 4):
        '''
        Global Environment for the Central Learning Router in decentral-qdina.
        The state represents the routing table, and actions modify query destinations.
        
        :param n_templates: Number of unique query templates (e.g., 22 for TPC-H)
        :param n_replicas: Total number of active database replicas in the cluster
        '''
        super(GlobalRoutingEnv, self).__init__()
        
        self.n_templates = n_templates
        self.n_replicas = n_replicas
        
        # Observation: routes (n_templates) + costs (n_templates) + worker loads (n_replicas)
        self.observation_space = gym.spaces.Box(
            low=0, high=np.inf,
            shape=(self.n_templates * 2 + self.n_replicas,)
        )
        self.n_actions = (self.n_templates * (self.n_replicas - 1)) + 1
        self.action_space = gym.spaces.Discrete(self.n_actions)
        
        self._state_routes = np.zeros(self.n_templates, dtype=np.int32)
        self._state_costs = np.zeros(self.n_templates, dtype=np.float64)
        self._state_worker_loads = np.zeros(self.n_replicas, dtype=np.int32)

    def _decode_action(self, action: int):
        '''
        Decodes the single discrete action integer into a routing instruction.
        Returns None for "Do Nothing", or (template_idx, target_replica_idx) for alterations.
        '''
        if action == 0:
            return None  # Action: Do Nothing
            
        # Shift index to simplify mathematical mapping
        adj_action = action - 1
        template_idx = adj_action // (self.n_replicas - 1)
        replica_shift = adj_action % (self.n_replicas - 1)
        current_replica = self._state_routes[template_idx]
        target_replica = replica_shift if replica_shift < current_replica else replica_shift + 1
        return template_idx, target_replica

    def _get_obs(self):
        return np.concatenate([
            self._state_routes,
            self._state_costs,
            self._state_worker_loads
        ])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options and 'initial_routing' in options:
            self._state_routes = np.array(options['initial_routing'], dtype=np.int32)
        else:
            # self._state_routes = np.random.randint(0, self.n_replicas, size=self.n_templates, dtype=np.int32)
            self._state_routes = np.array([i % self.n_replicas for i in range(self.n_templates)], dtype=np.int32)
        
        self._state_costs = np.zeros(self.n_templates, dtype=np.float64)
        self._state_worker_loads = np.zeros(self.n_replicas, dtype=np.int32)
        return self._get_obs(), {}

    def step(self, action: int, external_costs=None, external_template_costs=None, worker_loads=None):
        """
        Execute one global routing configuration change step.

        :param action: The chosen action.
        :param external_costs: List of total costs per replica (length n_replicas).
        :param external_template_costs: List of costs per template (length n_templates).
        :param worker_loads: List of number of queries per replica (length n_replicas), used for load balancing penalty.
        """
        old_routes = self._state_routes.copy()

        instruction = self._decode_action(action)
        if instruction is not None:
            template_idx, target_replica = instruction
            self._state_routes[template_idx] = target_replica

        if external_template_costs is not None:
            self._state_costs = np.log10(np.array(external_template_costs, dtype=np.float64) + 1.0)

        if external_costs is not None:
            costs = np.array(external_costs, dtype=np.float64)
        else:
            costs = np.zeros(self.n_replicas, dtype=np.float64)

        # Update worker loads if provided
        if worker_loads is not None:
            self._state_worker_loads = np.array(worker_loads, dtype=np.int32)

        makespan_raw = float(np.max(costs))
        sum_costs = np.sum(costs)
        sum_sq_costs = np.sum(costs ** 2)

        if sum_sq_costs > 0:
            jain_index = (sum_costs ** 2) / (self.n_replicas * sum_sq_costs)
        else:
            jain_index = 1.0

        makespan_scaled = np.log10(makespan_raw + 1.0)
        num_changes = np.sum(old_routes != self._state_routes)

        # Penalize load imbalance (standard deviation of worker loads)
        load_std = np.std(self._state_worker_loads) if self._state_worker_loads.size > 1 else 0.0
        load_penalty = 0.01 * load_std  # small penalty, can be tuned

        reward = -makespan_scaled + jain_index - 0.01 * num_changes - load_penalty

        if np.any(costs == 0.0) and np.sum(costs) > 0:
            reward -= 5.0

        info = {'makespan': makespan_raw, 'jain_index': jain_index, 'worker_loads': self._state_worker_loads}
        return self._get_obs(), reward, False, False, info