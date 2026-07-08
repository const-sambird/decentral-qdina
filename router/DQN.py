import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions, layer_features):
        super(DQN, self).__init__()
        
        # Use nn.Sequential for clean registration of weight keys (State Dict)
        modules = []
        modules.append(nn.Flatten())
        modules.append(nn.Linear(n_observations, layer_features[0]))
        modules.append(nn.ReLU())\
        
        for i in range(1, len(layer_features)):
            modules.append(nn.Linear(layer_features[i-1], layer_features[i]))
            modules.append(nn.ReLU())
            
        modules.append(nn.Linear(layer_features[-1], n_actions))
        
        self.network = nn.Sequential(*modules)
    
    def forward(self, x):
        return self.network(x)