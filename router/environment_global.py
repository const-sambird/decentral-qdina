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
        
        self.observation_space = gym.spaces.Box(
            low=0, high=np.inf,
            shape=(n_templates + n_templates + n_replicas + (n_templates * n_replicas),)
        )
        self.n_actions = (self.n_templates * (self.n_replicas - 1)) + 1
        self.action_space = gym.spaces.Discrete(self.n_actions)
        
        self._state_routes = np.zeros(self.n_templates, dtype=np.int32)
        self._state_costs = np.zeros(self.n_templates, dtype=np.float64)
        self._state_worker_loads = np.zeros(self.n_replicas, dtype=np.int32)

        self._state_cost_matrix = np.zeros((self.n_templates, self.n_replicas), dtype=np.float64)
        self._previous_makespan = None  # For reward shaping

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
        """Construct the observation vector for the global routing environment.

        The observation consists of:
        - Current routing table (template-to-replica mapping)
        - Aggregated costs per template (log10 normalized)
        - Worker loads (number of queries per replica)
        - Full cost matrix (per-template, per-replica costs)
        """
        # Flatten the cost matrix (row-major order)
        flat_matrix = self._state_cost_matrix.flatten()
        return np.concatenate([
            self._state_routes.astype(np.float32),
            self._state_costs.astype(np.float32),
            self._state_worker_loads.astype(np.float32),
            flat_matrix.astype(np.float32)
        ])

    def reset(self, seed=None, options=None):
        """Reset the global routing environment to its initial state.

        If 'initial_routing' is provided in options, it is used to set the routing table.
        Otherwise, routes are evenly distributed across replicas.

        Args:
            seed (int, optional): Random seed.
            options (dict, optional): May contain 'initial_routing' list.

        Returns:
            tuple: (observation, info)
        """
        super().reset(seed=seed)
        if options and 'initial_routing' in options:
            self._state_routes = np.array(options['initial_routing'], dtype=np.int32)
        else:
            self._state_routes = np.array([i % self.n_replicas for i in range(self.n_templates)], dtype=np.int32)
        
        self._state_costs = np.zeros(self.n_templates, dtype=np.float64)
        self._state_worker_loads = np.zeros(self.n_replicas, dtype=np.int32)
        self._state_cost_matrix.fill(0.0)
        self._previous_makespan = None
        return self._get_obs(), {}

    def step(self, action: int, external_costs=None, external_template_costs=None, worker_loads=None):
        """
        Execute one global routing configuration change step.

        :param action: The chosen action.
        :param external_costs: List of total costs per replica (length n_replicas).
        :param external_template_costs: If provided, can be:
            - A 2D array (n_templates, n_replicas) to fill the cost matrix.
            - A 1D array (n_templates) to fill only the aggregated costs.
        :param worker_loads: List of number of queries per replica (length n_replicas), used for load balancing penalty.
        """
        old_routes = self._state_routes.copy()

        instruction = self._decode_action(action)
        if instruction is not None:
            template_idx, target_replica = instruction
            self._state_routes[template_idx] = target_replica

        # Update the cost matrix if a 2D array is given
        if external_template_costs is not None:
            arr = np.array(external_template_costs)
            if arr.ndim == 2 and arr.shape == (self.n_templates, self.n_replicas):
                self._state_cost_matrix = arr.astype(np.float64)
                # Also update the aggregated costs (log10 normalized) from the matrix
                # Sum over replicas, then log10
                sum_per_template = np.sum(arr, axis=1)
                self._state_costs = np.log10(sum_per_template + 1.0)
            elif arr.ndim == 1 and len(arr) == self.n_templates:
                # Fallback: use the vector as aggregated costs (old behavior)
                self._state_costs = np.log10(arr.astype(np.float64) + 1.0)
                # Do not update the matrix, keep it as zeros
            # else: ignore unrecognized shape

        if external_costs is not None:
            costs = np.array(external_costs, dtype=np.float64)
        else:
            costs = np.zeros(self.n_replicas, dtype=np.float64)

        # Update worker loads if provided
        if worker_loads is not None:
            self._state_worker_loads = np.array(worker_loads, dtype=np.int32)

        makespan_raw = float(np.max(costs))
        makespan_scaled = np.log10(makespan_raw + 1.0)

        # Calculate Jain index for logging only (not used in reward)
        sum_costs = np.sum(costs)
        sum_sq_costs = np.sum(costs ** 2)
        if sum_sq_costs > 0:
            jain_index = (sum_costs ** 2) / (self.n_replicas * sum_sq_costs)
        else:
            jain_index = 1.0

        num_changes = np.sum(old_routes != self._state_routes)

        change_penalty = 0.1 * num_changes

        # Reward based on relative makespan improvement
        current_makespan = float(np.max(costs))

        # Scale to get reward between roughly -10 and +10 (since improvement_ratio ∈ [-1, 1])
        reward = 7.0 - (makespan_raw / 100_000_000.0) * (4.0 / 3.0)
        reward -= change_penalty
        reward = max(-30.0, min(10.0, reward))

        self._previous_makespan = current_makespan

        # Additional penalty if any replica has zero cost (inactive)
        if np.any(costs == 0.0) and np.sum(costs) > 0:
            reward -= 5.0

        info = {
            'makespan': makespan_raw,
            'jain_index': jain_index,
            'worker_loads': self._state_worker_loads
        }
        return self._get_obs(), reward, False, False, info