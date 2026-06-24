import torch.nn as nn
import torch.nn.functional as F

from encoding import StateEncoder

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions, layer_features):
        super(DQN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Flatten())
        self.layers.append(nn.Linear(n_observations, layer_features[0]))
        for i in range(2, len(layer_features)):
            self.layers.append(nn.Linear(layer_features[i-1], layer_features[i]))
        self.layers.append(nn.Linear(layer_features[len(layer_features)-1], n_actions))
    
    def forward(self, x):
        #print('initial state tensor', x)
        x = self.layers[0](x)
        #assert False
        #print('encoded state tensor', x)
        for layer in self.layers[1:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)
