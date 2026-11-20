
from .gcn import RoomGraphEncoder
from .dataset import RPlanDataset
from .outershell import ShipGeometryEncoder
from .cross_attention import ImageGraphFusion
from torch_geometric.data import Data
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class RoomGenEncoder(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.gcn_model = RoomGraphEncoder(input_dim=1, hidden_dim=64, output_dim=128)
        self.encoder = ShipGeometryEncoder(latent_dim=512)
        self.fusion = ImageGraphFusion()

    def forward(self, outershell_mask, node_features, edge_features):
        graph = Data(
            x=node_features,
            edge_index=edge_features
        )
        graph_features = self.gcn_model(graph.x, graph.edge_index)
        _, outtershell_features = self.encoder(outershell_mask)
        output = self.fusion(outtershell_features, graph_features)
        return output


if __name__ == "__main__":
    dataset = RPlanDataset()
    encoder = RoomGenEncoder()
    img, house_mask, room_sizes, neighboorhood_graph = dataset[1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img.permute(1,2,0), cmap="gray")
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(house_mask.squeeze(), cmap="gray")
    axes[1].set_title("House Mask")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

    print(room_sizes)
    print(neighboorhood_graph)
    print(f"img features: {img.shape}")
    print(f"house features: {house_mask.shape}")
    print(f"sizes features: {room_sizes.shape}")
    print(f"neighboorhood features: {neighboorhood_graph.shape}")
    cond = encoder(house_mask.unsqueeze(0), room_sizes, neighboorhood_graph)
    print(f"out features: {cond.shape}")

