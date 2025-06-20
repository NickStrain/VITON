import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from diffusers import StableDiffusionXLInpaintPipeline, DDPMScheduler,AutoencoderKL,UNet2DConditionModel
from transformers import CLIPImageProcessor,CLIPVisionModel, CLIPVisionModelWithProjection,CLIPTextModelWithProjection,CLIPTokenizer
import numpy as np
from PIL import Image

from diffusers import UNet2DModel 

class GarmentNet(nn.Module):
    def __init__(self, unet_model: UNet2DModel):
        super().__init__()
        self.encoder_blocks = unet_model.down_blocks
        self.mid_block = unet_model.mid_block
        self.in_conv = nn.Conv2d(in_channels=3,out_channels=128, kernel_size=3, bias=False)

    def forward(self, x):
        x = self.in_conv(x)
        features = []
        for block in self.encoder_blocks:
            x = block(x)
            features.append(x)
        x = self.mid_block(x)
        features.append(x)
        return features

class TryonNet(nn.Module):
    def __init__(self, unet_model: UNet2DModel):
        super().__init__()
        self.unet = unet_model
        self._expand_input_channels(8)  # 13 = 4 (person) + 3 (mask) + 3 (masked person) + 3 (densepose)
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    def _expand_input_channels(self, new_in_channels):
        old_conv = self.unet.conv_in
        self.unet.conv_in = nn.Conv2d(
            in_channels=new_in_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding
        )
        with torch.no_grad():
            self.unet.conv_in.weight[:, :old_conv.in_channels] = old_conv.weight
            self.unet.conv_in.weight[:, old_conv.in_channels:] = 0
            self.unet.conv_in.bias = old_conv.bias

    def forward(self, x, garment_features=None, clip_features=None):
        # x: concatenated person_lat, mask, masked_person, densepose
        noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
        timesteps = torch.randint(
                    0,
                    self.noise_scheduler.config.num_train_timesteps,
                    (4,),
                    dtype=torch.int64,
                    device=x.device
                    )

        return self.unet(x,timesteps).sample  # UNet2DModel returns an object with .sample


class IDM_VTON(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Use your small UNet2DModel
        base_unet = UNet2DModel(
            sample_size=config.image_size,
            in_channels=8,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=(
                "DownBlock2D", "DownBlock2D", "DownBlock2D",
                "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"
            ),
            up_block_types=(
                "UpBlock2D", "AttnUpBlock2D", "UpBlock2D",
                "UpBlock2D", "UpBlock2D", "UpBlock2D"
            )
        )

        garment_base_net =  UNet2DModel(
            sample_size=config.image_size,
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=(
                "DownBlock2D", "DownBlock2D", "DownBlock2D",
                "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"
            ),
            up_block_types=(
                "UpBlock2D", "AttnUpBlock2D", "UpBlock2D",
                "UpBlock2D", "UpBlock2D", "UpBlock2D"
            )
        )

        self.tryon_net = TryonNet(base_unet)
        self.garment_net = GarmentNet(garment_base_net)

        self.clip_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")

    def forward(self, person_lat, garment_lat, mask_lat, densepose, text_embeddings=None, caption=None):
        masked_person_lat = person_lat * (1 - mask_lat)
        x = torch.cat([person_lat, mask_lat, masked_person_lat, densepose], dim=1)
        # garment_feats = self.garment_net(garment_lat)
        clip_feats = self.encode_clip(garment_lat)
        output = self.tryon_net(x, garment_features=garment_lat, clip_features=clip_feats)
        return output

    def encode_clip(self, garment_rgb):
        inputs = self.clip_processor(images=garment_rgb, return_tensors="pt").to(garment_rgb.device)
        with torch.no_grad():
            output = self.clip_encoder(**inputs)
        return output.last_hidden_state