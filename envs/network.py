# network.py

import torch
import torch.nn as nn


#Shared network for policy and value functions
class SharedNetwork(nn.Module):
    def __init__(self, hidden_sizes):
        super(SharedNetwork, self).__init__()
        self.gru = nn.GRU(2, 256, num_layers=1, batch_first=True)
        dense_layers = []
        prev_dim = hidden_sizes[0]
        dense_layers.append(nn.Linear(256, 256))
        dense_layers.append(nn.ReLU())
 
        for hidden_size in hidden_sizes[1:]:
            dense_layers.append(nn.Linear(prev_dim, hidden_size))   
            dense_layers.append(nn.ReLU())
            prev_dim = hidden_size
        dense_layers.append(nn.Linear(prev_dim, 3))

        self.mlp = nn.Sequential(*dense_layers)     #dense layers

    def forward(self, x, state_in):
        gru_out, _ = self.gru(x, state_in)
        dense_out = self.mlp(gru_out[:, -1, :])
        return dense_out, gru_out