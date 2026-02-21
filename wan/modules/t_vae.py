import torch
import torch.nn as nn


from taehv import TAEHV

class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

class TAEW2_1DiffusersWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.dtype = torch.float16
        self.device = "cuda" if torch.cuda.is_available() else 'cpu'
        self.taehv = TAEHV("taew2_1.pth").to(self.dtype)
        self.temperal_downsample = [True, True, False] # [sic]
        self.config = DotDict(scaling_factor=1.0, latents_mean=torch.zeros(16), z_dim=16, latents_std=torch.ones(16))

    def decode(self, latents, return_dict=None):
        n, c, t, h, w = latents.shape
        # low-memory, set parallel=True for faster + higher memory
        return (self.taehv.decode_video(latents.transpose(1, 2), parallel=False).transpose(1, 2).mul_(2).sub_(1),)