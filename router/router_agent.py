import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from router.DQN import DQN

class RouterAgent:
    def __init__(self, n_templates: int, n_replicas: int, n_actions: int, layer_features: list = [64, 64], lr: float = 1e-3, gamma: float = 0.99):        
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
        
        input_size = n_templates * n_replicas
        self.n_replicas = n_replicas
        self.n_templates = n_templates

        # Pass input_size instead of n_templates
        self.policy_net = DQN(input_size, n_actions, layer_features)
        self.target_net = DQN(input_size, n_actions, layer_features)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target net remains in evaluation mode during training steps
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        
    def select_action(self, state, epsilon: float):
        '''
        Selects an action using the epsilon-greedy policy.
        
        :param state: The current global routing table state vector
        :param epsilon: Exploration probability threshold
        '''
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)
        
        with torch.no_grad():
            # Create tensor with 'long' type (64-bit integer)
            state_tensor = torch.tensor(state, dtype=torch.long)
            
            # Apply One-Hot Encoding
            one_hot_state = F.one_hot(state_tensor, num_classes=self.n_replicas)
            
            # Flatten and convert back to Float for the network
            one_hot_flat = one_hot_state.view(1, -1).float()
            
            # Move to target device dynamically
            device = next(self.policy_net.parameters()).device
            one_hot_flat = one_hot_flat.to(device)
            
            q_values = self.policy_net(one_hot_flat)
            return q_values.argmax().item()

    def learn(self, memory, batch_size):
        '''
        Optimizes the central router DQN policy net using standard Bellman Equation.
        '''
        if len(memory) < batch_size:
            return

        # Sample transitions from the replay buffer
        transitions = memory.sample(batch_size)
        
        states = [t.state for t in transitions]
        actions = [t.action for t in transitions]
        next_states = [t.next_state for t in transitions]
        rewards = [t.reward for t in transitions]

        # Determine target device (CPU/GPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Conversion to LONG type (required for one-hot encoding)
        state_b = torch.tensor(np.array(states), dtype=torch.long, device=device)
        next_state_b = torch.tensor(np.array(next_states), dtype=torch.long, device=device)
        action_b = torch.tensor(actions, dtype=torch.long, device=device).unsqueeze(1)
        reward_b = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)

        # Apply One-Hot Encoding and Flatten
        # Shape changes from [Batch, 22] -> [Batch, 22, 3] -> [Batch, 66]
        state_b = F.one_hot(state_b, num_classes=self.n_replicas).view(batch_size, -1).float()
        next_state_b = F.one_hot(next_state_b, num_classes=self.n_replicas).view(batch_size, -1).float()

        # Calculate current Q values: Q(s_t, a_t)
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