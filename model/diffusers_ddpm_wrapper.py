import torch
import torch.nn as nn
from diffusers import DDPMPipeline

class DiffusersDDPMWrapper(nn.Module):
    """
    Wraps HuggingFace DDPM model UNet for DAPS framework.
    Convers episilon prediction into x0hat prediction."""

    def __init__(self, model_id: str = "google/ddpm-celebahq-256"):
        super().__init__()
        pipe = DDPMPipeline.from_pretrained(model_id)
        self.unet = pipe.unet
        self.scheduler = pipe.scheduler  # diffusion scheduler
        self.in_channels = 3
        self.resolution = 256 # CelebA-HQ resolution
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self.device)

    def get_in_shape(self):
        """
        DAPS calls this to get random input xt shape.
        Returns:
            tuple: (C, H, W)
        """
        return (self.in_channels, self.resolution, self.resolution)
    
    def predict_eps(self, xt, t):
        """
        Use UNet to predict noise eps
        """
        return self.unet(xt, t).sample
    
    def predict_x0(self, xt, t):
        """
        Convert eps prediction to x0hat prediction
        """
        eps = self.predict_eps(xt, t)
        alpha_prod = self.scheduler.alphas_cumprod[t].to(self.device).reshape(-1, 1, 1, 1) # (B, 1, 1, 1)

        x0 = (xt - torch.sqrt(1 - alpha_prod) * eps) / torch.sqrt(alpha_prod)
        return x0
    
    # IMPORTANT: this is the API function that PF-ODE sampler will call
    def forward(self, xt, sigma_t, *args, **kwargs):
        """
        DAPS expects: model(xt, sigma_t) -> x0hat
        To do so, convert sigma_t to discrete timestep t
        """
        t = self.sigma_to_t(sigma_t).long()
        return self.predict_x0(xt, t)
    
    def sigma_to_t(self, sigma):
        """
        Diffusers use discrete timesteps, while DAPS uses continuous noise levels (sigma_t) at each step.
        """
        sigma = torch.sqrt(1 - self.scheduler.alphas_cumprod).to(sigma.device)  # (T,)

        d = (sigma - sigma.unsqueeze(0)).abs()
        t = d.argmin(dim=1)
        return t