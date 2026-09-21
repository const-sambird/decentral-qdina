import numpy as np
import random

class StaticRouterAgent:
    """
    Static (non-learned) router for the ablation study.

    This router does NOT use any neural network and does NOT learn.
    The routing table is fixed at the beginning of each episode by
    `initialize_routing_table()` (greedy load balancing), and this
    agent simply returns the "Do Nothing" action at every step.

    This corresponds to the "Decentral-DINA without learned routing"
    experiment in the ablation study.

    Interface-compatible with RouterAgent.
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

    def select_action(self, state, epsilon: float):
        """
        Always returns action 0 ("Do Nothing").
        The routing table is therefore never modified during an episode.

        :param state: The current state vector (ignored)
        :param epsilon: Exploration rate (ignored)
        :returns: 0 (Do Nothing)
        """
        return 0

    def learn(self, memory, batch_size):
        """No-op: this router does not learn."""
        pass

    def soft_update(self, tau: float = 0.005):
        """No-op: this router does not learn."""
        pass