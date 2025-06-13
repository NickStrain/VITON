import torch 
import torch.nn as nn 
from diffusers import StableDiffusionInpaintPipeline


class GarmentNet(nn.Module):
    '''
    GarmenNet model
    '''
    def __init__(self,base_unet):
        super().__init__()
        self.unet_encoders = nn.ModuleList(base_unet.down_blocks)
        self.unet_mid = base_unet.mid_block

    
    def foward(self,x):
        Garment_features = []
        for block in self.unet_encoders:
            x = block(x)
            Garment_features.append(x)
        x = self.unet_mid(x)
        Garment_features.append(x)

        return Garment_features

class TryonNet(nn.Module):
    '''
    TryonNet 
    ''' 
    def __init__(self,base_unet):
        super().__init__()
        self.base_unet = base_unet

    def forward(self,x,Garment_features):
        pass
    

class Viton(nn.Module):
    '''
    Final VTIB model 
    '''
    def __init__(self):
        super().__init__()

    def forward():
        pass 



pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16,
    safety_checker=None  # optional
).to("cuda")

print(pipe.unet)