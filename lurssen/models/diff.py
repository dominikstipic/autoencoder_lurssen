import torch
import torch.nn.functional as F
from torch import nn
from .unet import ConditionalUNet


class ConditionalDDPM:

    def __init__(
        self,
        model,
        timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        device="cuda"
    ):
        self.model = model
        self.timesteps = timesteps
        self.device = device

        # Linear noise schedule
        self.betas = torch.linspace(
            beta_start,
            beta_end,
            timesteps
        ).to(device)

        self.alphas = 1.0 - self.betas

        self.alpha_bars = torch.cumprod(
            self.alphas,
            dim=0
        )

        # alpha_bar at the previous timestep
        self.alpha_bars_prev = F.pad(
            self.alpha_bars[:-1],
            (1, 0),
            value=1.0
        )

        # Posterior variance
        self.posterior_variance = (
            self.betas
            * (1.0 - self.alpha_bars_prev)
            / (1.0 - self.alpha_bars)
        )

    # ---------------------------------------
    # Forward diffusion
    # ---------------------------------------

    def q_sample(self, x0, t, noise=None):

        if noise is None:
            noise = torch.randn_like(x0)

        alpha_bar = self.alpha_bars[t]

        alpha_bar = alpha_bar.view(-1, 1, 1, 1)

        # x_t = sqrt(alpha_bar) * x_0
        #       + sqrt(1 - alpha_bar) * noise

        xt = (
            torch.sqrt(alpha_bar) * x0
            + torch.sqrt(1.0 - alpha_bar) * noise
        )

        return xt, noise

    # ---------------------------------------
    # Training loss
    # ---------------------------------------

    def loss(self, x0, condition):

        batch_size = x0.shape[0]

        # Random diffusion timesteps
        t = torch.randint(
            0,
            self.timesteps,
            (batch_size,),
            device=self.device
        )

        # Sample Gaussian noise
        noise = torch.randn_like(x0)

        # Generate noisy image
        xt, noise = self.q_sample(
            x0,
            t,
            noise
        )

        # Convert timesteps to float for time embedding
        t_float = t.float()

        # Predict the added noise
        predicted_noise = self.model(
            xt,
            t_float,
            condition
        )

        # DDPM noise-prediction objective
        return F.mse_loss(
            predicted_noise,
            noise
        )

    # ---------------------------------------
    # Reverse diffusion / sampling
    # ---------------------------------------

    @torch.no_grad()
    def sample(
        self,
        condition,
        image_size=256,
        channels=1
    ):

        self.model.eval()

        batch_size = condition.shape[0]

        # Start from Gaussian noise
        x = torch.randn(
            batch_size,
            channels,
            image_size,
            image_size,
            device=self.device
        )

        # Reverse diffusion: T-1 -> 0
        for t in reversed(range(self.timesteps)):

            t_batch = torch.full(
                (batch_size,),
                t,
                device=self.device,
                dtype=torch.float32
            )

            # Predict noise
            predicted_noise = self.model(
                x,
                t_batch,
                condition
            )

            alpha_t = self.alphas[t]
            alpha_bar_t = self.alpha_bars[t]
            beta_t = self.betas[t]

            # DDPM reverse-process mean
            mean = (
                1.0 / torch.sqrt(alpha_t)
            ) * (
                x
                - (beta_t / torch.sqrt(1.0 - alpha_bar_t))
                * predicted_noise
            )

            # Add noise except at final timestep
            if t > 0:

                noise = torch.randn_like(x)

                variance = self.posterior_variance[t]

                x = mean + torch.sqrt(variance) * noise

            else:
                x = mean

        return x

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ConditionalUNet(
        in_channels=1,
        cond_channels=512,
        base_channels=32,
        time_dim=256
    ).to(device)
    diffusion = ConditionalDDPM(
        model,
        timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        device=device
    )
    model.eval()
    condition = torch.randn(
        1, 512, 8, 8,
        device=device
    )

    with torch.no_grad():
        generated_floorplan = diffusion.sample(
            condition,
            image_size=256,
            channels=1
        )
    print("Generated floor plan:", generated_floorplan.shape)
