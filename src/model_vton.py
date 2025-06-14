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

        old_conv = self.base_unet.conv_in
        new_conv = nn.Conv2d(13,old_conv.out_channels, old_conv.kernel_size, old_conv.stride, old_conv.padding)

        with torch.no_grad():
            new_conv.weight[:,:old_conv.in_channels] = old_conv.weight
            new_conv.weight[:,old_conv.in_channels:] = 0
            new_conv.bias[:] = old_conv.bias

    def forward(self,latents,garment_features):
        h = latents 
        skips = []

        for block in self.base_unet.down_blocks():
            h = block(h)
            skips.append(h)
        h = self.base_unet.mid_block(h)

        for i, block in enumerate(self.base_unet.up_blocks):
            skip = skips.pop()
            g = garment_features[-(i + 1)]
            h = torch.cat([h, skip, g], dim=1)
            h = block(h)

        return self.base_unet.conv_norm_out(self.base_unet.conv_out(h))
            
        
    

class Viton(nn.Module):
    '''
    Final VTIB model 
    '''
    def __init__(self):
        super().__init__()
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            torch_dtype=torch.float16,
            safety_checker=None  # optional
            ).to("cuda")
        self.tryonnet = TryonNet(pipe.unet)
        self.garmentnet = GarmentNet(pipe.unet)

    def forward(self,
                person_lat, mask_lat, masked_person_lat, pose_lat,
                garment_rgb):
        garment_feats = self.garmentnet(garment_rgb)
        x = torch.cat([person_lat, mask_lat, masked_person_lat, pose_lat], dim=1)
        x = x.to("cuda")

        return self.tryonnet(x, garment_feats) 



# pipe = StableDiffusionInpaintPipeline.from_pretrained(
#     "stabilityai/stable-diffusion-2-inpainting",
#     torch_dtype=torch.float16,
#     safety_checker=None  # optional
# ).to("cuda")

# print(pipe.unet)

obj = Viton()
print(obj)