import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from router.DQN import DQN

class RouterAgent:
    def __init__(self, n_templates: int, n_replicas: int, n_actions: int, layer_features: list = [128, 128], lr: float = 1e-3, gamma: float = 0.99):
        '''
        Central Router Agent handling the global query distribution policy.
        
        :param n_templates: Number of unique query templates (input state size)
        :param n_replicas: Total number of active database replicas in the cluster
        :param n_actions: Total number of discrete routing actions
        :param layer_features: List containing hidden layer dimensions for the DQN
        :param lr: Learning rate for the optimizer
        :param gamma: Discount factor for long-term rewards
        '''
        self.n_actions = n_actions
        self.gamma = gamma
        self.n_replicas = n_replicas
        self.n_templates = n_templates
        
        # Input size: routes one-hot (n_templates * n_replicas) + costs (n_templates) + worker loads (n_replicas)
        input_size = (self.n_templates * self.n_replicas) + self.n_templates + self.n_replicas

        self.policy_net = DQN(input_size, n_actions, layer_features)
        self.target_net = DQN(input_size, n_actions, layer_features)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target net remains in evaluation mode during training steps
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def _prepare_state_tensor(self, state_batch, batch_size, device):
        """
        Prepares the state tensor for input into the DQN.
        State format: [routes (n_templates), costs (n_templates), worker_loads (n_replicas)]
        """
        # Extract routes (first n_templates elements)
        routes_raw = state_batch[:, :self.n_templates].to(torch.long)
        # Extract costs (next n_templates elements)
        costs_raw = state_batch[:, self.n_templates:2*self.n_templates].to(torch.float32)
        # Extract worker loads (last n_replicas elements)
        worker_loads_raw = state_batch[:, 2*self.n_templates:2*self.n_templates+self.n_replicas].to(torch.float32)
        
        routes_one_hot = F.one_hot(routes_raw, num_classes=self.n_replicas).view(batch_size, -1).float()
        
        return torch.cat([routes_one_hot, costs_raw, worker_loads_raw], dim=1)
        
    def select_action(self, state, epsilon: float):
        '''
        Selects an action using the epsilon-greedy policy.
        
        :param state: The current global routing table state vector
        :param epsilon: Exploration probability threshold
        '''
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)
        
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            device = next(self.policy_net.parameters()).device
            nn_input = self._prepare_state_tensor(state_tensor, 1, device)
            
            q_values = self.policy_net(nn_input)
            return q_values.argmax().item()

    def learn(self, memory, batch_size):
        if len(memory) < batch_size:
            return

        # Sample transitions from the replay buffer
        transitions = memory.sample(batch_size)
        
        states = np.array([t.state for t in transitions])
        actions = np.array([t.action for t in transitions])
        next_states = np.array([t.next_state for t in transitions])
        rewards = np.array([t.reward for t in transitions])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        state_b_raw = torch.tensor(states, dtype=torch.float32)
        next_state_b_raw = torch.tensor(next_states, dtype=torch.float32)
        action_b = torch.tensor(actions, dtype=torch.long, device=device).unsqueeze(1)
        reward_b = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)

        state_b = self._prepare_state_tensor(state_b_raw, batch_size, device)
        next_state_b = self._prepare_state_tensor(next_state_b_raw, batch_size, device)

        current_q_values = self.policy_net(state_b).gather(1, action_b)

        # Calculate maximum future Q value: max_a Q_target(s_{t+1}, a)
        max_next_q_values = self.target_net(next_state_b).max(1)[0].unsqueeze(1)
        
        # Bellman equation for expected target value (Isolate from gradient graph using .detach())
        expected_q_values = reward_b + (self.gamma * max_next_q_values.detach())

        # Calculate MSE loss and backpropagate gradient
        loss = self.loss_fn(current_q_values, expected_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        
        # Prevent gradient explosion from massive costs
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        
        self.optimizer.step()

    def soft_update(self, tau=0.005):
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1 - tau) * target_param.data)