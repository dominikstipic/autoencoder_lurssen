import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from lurssen.models.unet import ConditionalUNet

class ConditionalFlowMatching:

    def __init__(self, model, device="cuda"):

        self.model = model
        self.device = device

    def loss(self, x1, condition):

        batch_size = x1.shape[0]

        # Gaussian noise
        x0 = torch.randn_like(x1)

        # Random time
        t = torch.rand(
            batch_size,
            device=self.device
        )

        # Reshape time
        t_view = t.view(
            batch_size, 1, 1, 1
        )

        # Linear interpolation
        xt = (1 - t_view) * x0 + t_view * x1

        # Target velocity
        target_velocity = x1 - x0

        # Predicted velocity
        predicted_velocity = self.model(
            xt,
            t,
            condition
        )

        return F.mse_loss(
            predicted_velocity,
            target_velocity
        )

    @torch.no_grad()
    def sample(
        self,
        condition,
        image_size=256,
        channels=1,
        steps=50
    ):

        self.model.eval()

        batch_size = condition.shape[0]

        # Initial Gaussian noise
        x = torch.randn(
            batch_size,
            channels,
            image_size,
            image_size,
            device=self.device
        )

        dt = 1.0 / steps

        # Euler ODE integration
        for i in range(steps):

            t = torch.full(
                (batch_size,),
                i / steps,
                device=self.device
            )

            velocity = self.model(
                x,
                t,
                condition
            )

            x = x + dt * velocity

        return x

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ConditionalUNet(
        in_channels=1,
        cond_channels=512,
        base_channels=32,
        time_dim=256
    ).to(device)

    flow = ConditionalFlowMatching(
        model,
        device=device
    )

    B = 1
    x_t = torch.randn(B, 1, 256, 256).to(device)
    cond = torch.randn(B, 512, 8, 8).to(device)
    t = torch.rand(B).to(device)
    with torch.no_grad():
        velocity = model(x_t, t, cond)

    print("Input shape:", x_t.shape)
    print("Condition shape:", cond.shape)
    print("Time shape:", t.shape)
    print("Output shape:", velocity.shape)

