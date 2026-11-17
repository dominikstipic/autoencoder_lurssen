import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv

class RoomGraphEncoder(nn.Module):

    def __init__(self, input_dim=8, hidden_dim=64, output_dim=128):
        super().__init__()

        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        return x