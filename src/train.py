import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator
from torchvision import transforms
import wandb
from diffusers.optimization import get_cosine_schedule_with_warmup
# from model_vton import Viton
from preprocess import DatasetLoader
import lpips
from model_vton import IDM_VTON
from dataclasses import dataclass
from accelerate import Accelerator
from tqdm.auto import tqdm

@dataclass
class TrainingConfig:
    image_size = 128 
    train_batch_size = 16
    eval_batch_size = 16  
    num_epochs = 50
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 30
    mixed_precision = "fp16"  
    seed = 0

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),])
train_dataset = DatasetLoader("../train",transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True,num_workers=0)

config = TrainingConfig()
model = IDM_VTON(config)


optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)

accelerator = Accelerator(mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        )
model.train()
global_step = 0
for epoch in range(config.num_epochs):
    progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
    progress_bar.set_description(f"Epoch {epoch}")

    for step, batch in enumerate(train_dataloader):
        person_lat = batch["person_lat"].to(accelerator.device)
        mask_lat = batch["mask_lat"].to(accelerator.device)
        masked_person_lat = batch["masked_person_lat"].to(accelerator.device)
        pose_lat = batch["pose_lat"].to(accelerator.device)
        garment_lat = batch["garment_lat"].to(accelerator.device)
       

        with accelerator.accumulate(model):
            output_latents = model(
                person_lat, garment_lat, mask_lat, pose_lat
            )
            decoded_imgs = model.pipeline.decode_latents(output_latents)

            loss_l1 = F.l1_loss(decoded_imgs, person_lat)
            loss_perceptual = model.perceptual_loss(decoded_imgs, person_lat)
            loss = loss_l1 + 0.8 * loss_perceptual

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        logs = {"loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
        progress_bar.set_postfix(**logs)
        accelerator.log(logs, step=global_step)
        global_step += 1
        progress_bar.update(1)
