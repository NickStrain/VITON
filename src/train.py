import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator
from torchvision import transforms
import wandb

from model_vton import Viton
from preprocess import DatasetLoader
import lpips


DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
def train():

    

    accelerator = Accelerator(mixed_precision='fp16')
    model = Viton()
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=1e-5, 
        weight_decay=0.01
    )
    l1_loss = nn.L1Loss()
    # percep_loss = lpips.LPIPS(net='vgg').to(DEVICE)

    transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),])
    train_dataset = DatasetLoader("../train",transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True,num_workers=0)
    
    model = Viton().to(DEVICE)
    model.train()

    model, optimizer, dataloader = accelerator.prepare(
        model, optimizer, dataloader
    )
    NUM_EPOCHS = 5
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print("SDFsd")
        for batch in pbar:
        # Unpack batch data
            person_lat = batch["person_lat"].to(DEVICE)            # (B, 4, H, W)
            mask_lat = batch["mask_lat"].to(DEVICE)                    # (B, 3, H, W)
            masked_person_lat = batch["masked_person_lat"].to(DEVICE)  # (B, 3, H, W)
            pose_lat = batch["pose_lat"].to(DEVICE)               # (B, 3, H, W)
            garment_lat = batch["garment_rgb"].to(DEVICE)                                     # Optional if needed by tokenizer
           
            output = model(
                person_lat, garment_lat, mask_lat, pose_lat,
            )
            print(output)
    
if __name__ == "__main__":
    train()
