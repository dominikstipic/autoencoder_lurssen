
from lurssen.gcn import RoomGraphEncoder
from torch_geometric.data import Data
from lurssen.dataset import RPlanDataset
from lurssen.encoder import ShipGeometryEncoder
from lurssen.cross_attention import ImageGraphFusion
import torch.nn as nn
import torch
from torch.utils.data import DataLoader


class RoomGenModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.gcn_model = RoomGraphEncoder(input_dim=1, hidden_dim=64, output_dim=128)
        self.encoder = ShipGeometryEncoder(latent_dim=512)
        self.fusion = ImageGraphFusion()

# [1, 512, 8, 8]

    def forward(self, outershell_mask, node_features, edge_features):
        graph = Data(
            x=node_features,
            edge_index=edge_features
        )
        graph_features = self.gcn_model(graph.x, graph.edge_index)
        _, outtershell_features = self.encoder(outershell_mask)
        output = self.fusion(outtershell_features, graph_features)
        return output


dataset = RPlanDataset()

model = RoomGenModel()

img, house_mask, sizes, neighboorhood = dataset[1]
print(f"img features: {img.shape}")
print(f"house features: {house_mask.shape}")
print(f"sizes features: {sizes.shape}")
print(f"neighboorhood features: {neighboorhood.shape}")
out = model(house_mask.unsqueeze(0), sizes, neighboorhood)
print(f"out features: {out.shape}")

