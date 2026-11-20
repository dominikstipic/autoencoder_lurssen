import torch
from lurssen.models.unet import ConditionalUNet
from lurssen.models.diff import ConditionalDDPM

device = "cuda" if torch.cuda.is_available() else "cpu"

model = ConditionalUNet(
    in_channels=3,
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

B = 1

x0 = torch.randn(
    B, 3, 256, 256,
    device=device
)

condition = torch.randn(
    B, 512, 8, 8,
    device=device
)

t = torch.randint(
    0, 1000,
    (B,),
    device=device
)

with torch.no_grad():

    xt, noise = diffusion.q_sample(x0, t)

    predicted_noise = model(
        xt,
        t.float(),
        condition
    )

print("Original image:", x0.shape)
print("Noisy image:", xt.shape)
print("Condition:", condition.shape)
print("Timestep:", t.shape)
print("Actual noise:", noise.shape)
print("Predicted noise:", predicted_noise.shape)