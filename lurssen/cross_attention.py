import torch
import torch.nn as nn
import math

class CrossAttention(nn.Module):

    def __init__(self, query_dim, context_dim, hidden_dim):
        super().__init__()

        self.query = nn.Linear(query_dim, hidden_dim)
        self.key = nn.Linear(context_dim, hidden_dim)
        self.value = nn.Linear(context_dim, hidden_dim)

        self.output = nn.Linear(hidden_dim, query_dim)

    def forward(self, x_query, x_context):

        # Project inputs
        Q = self.query(x_query)
        K = self.key(x_context)
        V = self.value(x_context)

        # Calculate attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1))

        scores = scores / math.sqrt(K.shape[-1])

        # Normalize attention scores
        attention = torch.softmax(scores, dim=-1)

        # Weighted sum of values
        out = torch.matmul(attention, V)

        return self.output(out)

class ImageGraphFusion(nn.Module):

    def __init__(self):
        super().__init__()

        self.cross_attention = CrossAttention(
            query_dim=512,
            context_dim=128,
            hidden_dim=256
        )

        self.norm = nn.LayerNorm(512)

    def forward(self, image_features, graph_features):

        # Image: [B, 512, 8, 8]
        # Graph: [B, 7, 128]

        B, C, H, W = image_features.shape

        # Convert image into sequence
        image = image_features.flatten(2).transpose(1, 2)

        # Cross attention
        fused = self.cross_attention(image, graph_features)

        # Residual connection + normalization
        fused = self.norm(image + fused)

        # Restore spatial dimensions
        fused = fused.transpose(1, 2).reshape(B, C, H, W)

        return fused


