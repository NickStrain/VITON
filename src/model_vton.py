import torch 
import torch.nn as nn 
from diffusers import StableDiffusionXLInpaintPipeline


class GarmenNet(nn.Module):
    '''
    GarmenNet model
    '''
    def __init__(self,base_unet):
        super().__init__()

    
    def foward(self):
        pass

class TryonNet(nn.Module):
    '''
    TryonNet 
    ''' 
    def __init__(self,base_unet):
        super().__init__()
    
    def forward(self):
        pass
    

class Viton(nn.Module):
    '''
    Final VTIB model 
    '''
    def __init__(self):
        super().__init__()

    def forward():
        pass 



diffuser_model = StableDiffusionXLInpaintPipeline.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
        )

print(diffuser_model)