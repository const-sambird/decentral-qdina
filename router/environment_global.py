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
        
        # State Space (S): Routing table mapping each template to a replica ID (0 to n_replicas-1)
        # Bounded discretely to satisfy the Markov property without continuous complexity
        self.observation_space = gym.spaces.MultiDiscrete([self.n_replicas] * self.n_templates)
        
        # Action Space (A): Size = (|QT| * (n - 1)) + 1
        # Action 0: "Do Nothing" (stabilize the cluster)
        # Actions 1 to (|QT| * (n - 1)): Reroute template j to a different replica
        self.n_actions = (self.n_templates * (self.n_replicas - 1)) + 1
        self.action_space = gym.spaces.Discrete(self.n_actions)
        
        # Internal state allocation: [replica_for_Q0, replica_for_Q1, ..., replica_for_Q21]
        self._state = np.zeros(self.n_templates, dtype=np.int32)
        
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
        
        # Determine target node while skipping the current node assignment
        current_replica = self._state[template_idx]
        target_replica = replica_shift if replica_shift < current_replica else replica_shift + 1
        
        return template_idx, target_replica

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initial state setup: can be initialized via a best-fit heuristic or uniformly
        if options and 'initial_routing' in options:
            self._state = np.array(options['initial_routing'], dtype=np.int32)
        else:
            self._state = np.random.randint(0, self.n_replicas, size=self.n_templates, dtype=np.int32)
            
        return self._state, {}

    def step(self, action: int, external_costs=None):
        """
        Execute one global routing configuration change step.
        """
        instruction = self._decode_action(action)
        if instruction is not None:
            template_idx, target_replica = instruction
            self._state[template_idx] = target_replica

        # If external costs are provided by the gRPC server, use them
        if external_costs is not None:
            costs = external_costs
        else:
            # Fallback or default calculation if absent
            costs = np.zeros(self.n_replicas, dtype=np.float32)

        # Calculate Jain Index and Makespan
        makespan = float(np.max(costs))
        
        sum_costs = np.sum(costs)
        sum_sq_costs = np.sum(costs ** 2)
        
        if sum_sq_costs > 0:
            jain_index = (sum_costs ** 2) / (self.n_replicas * sum_sq_costs)
        else:
            jain_index = 1.0
            
        reward = -makespan * (2.0 - jain_index)
        
        if np.any(costs == 0.0) and np.sum(costs) > 0:
            reward -= 5_000_000_000.0  # Fixed penalty
            
        terminated = False
        truncated = False
        info = {
            'makespan': makespan,
            'jain_index': jain_index
        }
        
        # Returns standard Gymnasium format
        return self._state, reward, terminated, truncated, info