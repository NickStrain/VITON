import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from diffusers import StableDiffusionXLInpaintPipeline, DDPMScheduler,AutoencoderKL,UNet2DConditionModel
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection,CLIPTextModelWithProjection,CLIPTokenizer
import numpy as np
from PIL import Image

class IPAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip_image_processor = CLIPImageProcessor()
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        )
        # Freeze CLIP
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        
        # Projection layers
        self.image_proj = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.GELU(),
            nn.Linear(2048, 2048))
        
    def forward(self, garment_image):
        # Preprocess and encode garment image
        clip_image = self.clip_image_processor(garment_image, return_tensors="pt").pixel_values
        image_embeds = self.image_encoder(clip_image).image_embeds
        return self.image_proj(image_embeds)

class GarmentNet(nn.Module):
    def __init__(self, pretrained_path="stabilityai/stable-diffusion-2-inpainting"):
        super().__init__()
        # Use SDXL UNet encoder blocks
        sdxl_unet = UNet2DConditionModel.from_pretrained(pretrained_path)
        self.down_blocks = nn.ModuleList(sdxl_unet.down_blocks)
        self.mid_block = sdxl_unet.mid_block
        
        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, garment_latent):
        features = []
        x = garment_latent
        for down_block in self.down_blocks:
            x = down_block(x)
            features.append(x)
        features.append(self.mid_block(x))
        return features

class TryonNet(nn.Module):
    def __init__(self, base_unet, ip_adapter):
        super().__init__()
        self.unet = base_unet
        self.ip_adapter = ip_adapter

        # Expand input conv to 13 channels
        old_conv = self.unet.conv_in
        new_conv = nn.Conv2d(
            13, 
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding
        )
        
        with torch.no_grad():
            # Copy original weights (first 4 channels)
            new_conv.weight[:, :4] = old_conv.weight.clone()
            
            # Zero-init new channels (positions 4-12)
            new_conv.weight[:, 4:] = 0  
            
            # Copy original bias
            new_conv.bias = nn.Parameter(old_conv.bias.clone())
        
        self.unet.conv_in = new_conv
        
        # Inject garment feature fusion
        self._inject_fusion_layers()
    
    def _inject_fusion_layers(self):
        """Add fusion layers to self-attention modules"""
        for name, module in self.unet.named_modules():
            if "attentions" in name and "self" in name:
                # Create fusion layer
                fusion_layer = nn.Sequential(
                    nn.Conv2d(module.in_channels * 2, module.in_channels, 3, padding=1),
                    nn.ReLU()
                )
                setattr(module, 'fusion_layer', fusion_layer)
    
    def forward(self, latents, garment_feats, clip_embeds, timestep):
        # Process through UNet with fusion
        h = latents
        skips = []
        
        # Down blocks
        for i, down_block in enumerate(self.unet.down_blocks):
            h = down_block(h, timestep)
            skips.append(h)
            
            # Fuse garment features
            if i < len(garment_feats) - 1:  # -1 for mid block
                h = self._fuse_features(h, garment_feats[i], i)
        
        # Mid block
        h = self.unet.mid_block(h, timestep)
        h = self._fuse_features(h, garment_feats[-1], "mid")
        
        # Up blocks
        for i, up_block in enumerate(self.unet.up_blocks):
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            h = up_block(h, timestep)
            
            # Fuse garment features in later up blocks
            if i >= len(self.unet.up_blocks) - 2:  # Last two blocks
                g_idx = len(garment_feats) - i - 2
                h = self._fuse_features(h, garment_feats[g_idx], f"up_{i}")
        
        # Apply cross-attention with IP-Adapter
        return self._apply_cross_attention(h, clip_embeds)
    
    def _fuse_features(self, tryon_feat, garment_feat, block_name):
        """Fuse TryonNet and GarmentNet features"""
        # Attention fusion
        for name, module in self.unet.named_modules():
            if "attentions" in name and block_name in name:
                # Prepare inputs
                concat = torch.cat([tryon_feat, garment_feat], dim=1)
                
                # Apply fusion
                fused = module.fusion_layer(concat)
                return fused + tryon_feat  # Residual connection
        return tryon_feat
    
    def _apply_cross_attention(self, x, clip_embeds):
        """Apply cross-attention using IP-Adapter embeddings"""
        # Flatten spatial dimensions
        x_flat = x.flatten(2).permute(0, 2, 1)
        
        # Compute attention
        attn_scores = torch.matmul(x_flat, clip_embeds.transpose(1, 2)) / np.sqrt(x.shape[1])
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        # Apply attention
        attn_output = torch.matmul(attn_weights, clip_embeds)
        attn_output = attn_output.permute(0, 2, 1).view_as(x)
        
        return x + attn_output

class IDM_VTON(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        # Load pretrained SDXL inpainting model
        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/stable-diffusion-base-1.0", subfolder="vae")
        self.tokenizer = CLIPTokenizer.from_pretrained(
            "stabilityai/stable-diffusion-base-1.0", subfolder="tokenizer")
        self.text_encoder = CLIPTextModelWithProjection.from_pretrained(
            "stabilityai/stable-diffusion-base-1.0", subfolder="text_encoder")
        
        # Model components
        base_unet = UNet2DConditionModel.from_pretrained(
            "stabilityai/stable-diffusion-base-1.0", subfolder="unet")
        self.ip_adapter = IPAdapter()
        self.garment_net = GarmentNet()
        self.tryon_net = TryonNet(base_unet, self.ip_adapter)
        
        # Diffusion scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear"
        )
        
        self.device = device
        self.to(device)
    
    def encode_image(self, image):
        """Convert image to latent space"""
        if isinstance(image, Image.Image):
            image = np.array(image).transpose(2, 0, 1)
            image = torch.from_numpy(image).float() / 127.5 - 1.0
        return self.vae.encode(image.unsqueeze(0).latent_dist.sample()) * 0.18215
    
    def forward(self, batch):
        # Unpack batch
        person_img = batch["person_img"].to(self.device)
        garment_img = batch["garment_img"].to(self.device)
        mask = batch["mask"].to(self.device)
        pose = batch["pose"].to(self.device)
        caption = batch["caption"]
        
        # Encode inputs to latent space
        with torch.no_grad():
            person_latent = self.encode_image(person_img)
            garment_latent = self.encode_image(garment_img)
            pose_latent = self.encode_image(pose)
            masked_latent = person_latent * (1 - mask)
        
        # Prepare TryonNet input (13 channels)
        tryon_input = torch.cat([
            person_latent, 
            mask, 
            masked_latent, 
            pose_latent
        ], dim=1)
        
        # Get IP-Adapter embeddings
        clip_embeds = self.ip_adapter(garment_img)
        
        # Get GarmentNet features
        garment_feats = self.garment_net(garment_latent)
        
        # Encode text prompts
        text_inputs = self.tokenizer(
            caption, 
            padding="max_length", 
            max_length=77, 
            return_tensors="pt"
        ).to(self.device)
        text_embeds = self.text_encoder(**text_inputs).last_hidden_state
        
        # Sample noise and timesteps
        noise = torch.randn_like(person_latent)
        timesteps = torch.randint(
            0, self.noise_scheduler.num_train_timesteps, 
            (person_img.shape[0],), device=self.device
        ).long()
        
        # Add noise to target
        noisy_target = self.noise_scheduler.add_noise(person_latent, noise, timesteps)
        
        # Predict noise residual
        noise_pred = self.tryon_net(
            tryon_input,
            garment_feats,
            clip_embeds,
            timesteps
        )
        
        # Calculate loss
        loss = nn.functional.mse_loss(noise_pred, noise)
        return loss 