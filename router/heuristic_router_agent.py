import numpy as np
import random

class HeuristicRouterAgent:
    """
    Heuristic router for the ablation study.

    This router replaces the learned DQN with the routing heuristic used
    in DiversityClusterDB (Hang et al., 2024). The heuristic is rule-based
    and does NOT use any neural network or learning.

    This corresponds to the "Decentral-DINA with the DiversityClusterDB
    routing heuristic" experiment in the ablation study.

    Interface-compatible with RouterAgent.

    NOTE: The exact heuristic from Hang et al. is not reproduced here.
    A placeholder implementation is provided below. Replace the body of
    `select_action()` with the actual heuristic from the paper.
    """

    def __init__(self, n_templates: int, n_replicas: int, n_actions: int,
                 layer_features: list = None, lr: float = 0.0, gamma: float = 0.0):
        '''
        :param n_templates: Number of unique query templates
        :param n_replicas: Number of active database replicas
        :param n_actions: Size of the discrete action space
        :param layer_features: Ignored (kept for interface compatibility)
        :param lr: Ignored
        :param gamma: Ignored
        '''
        self.n_templates = n_templates
        self.n_replicas = n_replicas
        self.n_actions = n_actions

        # === Placeholder state ===
        # The heuristic needs to know the current routing table and costs.
        # Since `select_action` only receives the flat state vector, we
        # decode the relevant parts here.
        # State layout (see GlobalRoutingEnv._get_obs):
        #   [0 : n_templates]                            -> routes
        #   [n_templates : 2*n_templates]                -> costs (log10)
        #   [2*n_templates : 2*n_templates + n_replicas] -> worker loads
        #   [2*n_templates + n_replicas : ]              -> cost matrix (flattened)
        self._state_size_routes = n_templates
        self._state_size_costs = n_templates
        self._state_size_loads = n_replicas

    def _decode_state(self, state):
        """Extract routes, costs and loads from the flat state vector."""
        routes = state[:self._state_size_routes].astype(int)
        costs = state[self._state_size_routes:
                      self._state_size_routes + self._state_size_costs]
        loads = state[self._state_size_routes + self._state_size_costs:
                      self._state_size_routes + self._state_size_costs + self._state_size_loads]
        return routes, costs, loads

    def select_action(self, state, epsilon: float):
        """
        Select a routing action using the DiversityClusterDB heuristic.

        TODO: implement the actual heuristic from Hang et al. (2024).

        Placeholder logic: pick the template whose cost is the highest,
        and move it to the replica with the lowest current load.

        :param state: The current state vector.
        :param epsilon: Ignored (heuristic is deterministic).
        :returns: A discrete action index, or 0 (Do Nothing).
        """
        routes, costs, loads = self._decode_state(state)

        # Identify the most expensive template
        worst_template = int(np.argmax(costs))

        # Identify the least loaded replica
        best_replica = int(np.argmin(loads))

        # If the worst template is already on the best replica, do nothing
        if routes[worst_template] == best_replica:
            return 0

        # Encode the action (template_idx, target_replica) into a single integer.
        # See GlobalRoutingEnv._decode_action for the inverse mapping:
        #   action = 1 + template_idx * (n_replicas - 1) + replica_shift
        # where replica_shift is the index of the target replica in the list
        # of replicas excluding the current one.
        current = routes[worst_template]
        replica_shift = best_replica if best_replica < current else best_replica - 1
        action = 1 + worst_template * (self.n_replicas - 1) + replica_shift
        return action

    def learn(self, memory, batch_size):
        """No-op: this router does not learn."""
        pass

    def soft_update(self, tau: float = 0.005):
        """No-op: this router does not learn."""
        pass