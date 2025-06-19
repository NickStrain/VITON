import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator
from torchvision import transforms
import wandb

# from model_vton import Viton
# from preprocess import DatasetLoader
import lpips


# DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
# def train():

#     # accelerator = Accelerator(mixed_precision='fp16')
#     model = Viton()
#     optimizer = torch.optim.AdamW(
#         model.parameters(), 
#         lr=1e-5, 
#         weight_decay=0.01
#     )
#     l1_loss = nn.L1Loss()
#     # percep_loss = lpips.LPIPS(net='vgg').to(DEVICE)

#     transform = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor(),])
#     train_dataset = DatasetLoader("../train",transform=transform)
#     train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    
#     model.train()

#     # model, optimizer, train_loader = accelerator.prepare(
#     #     model, optimizer, train_loader
#     # )
#     NUM_EPOCHS = 5
#     for epoch in range(NUM_EPOCHS):
#         epoch_loss = 0
#         pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
#         print("SDFsd")
#         for batch in pbar:
#         # Unpack batch data
#             person_lat = batch["person_lat"]            # (B, 4, H, W)
#             mask_lat = batch["mask_lat"]                    # (B, 3, H, W)
#             masked_person_lat = batch["masked_person_lat"]  # (B, 3, H, W)
#             pose_lat = batch["pose_lat"]               # (B, 3, H, W)
#             garment_lat = batch["garment_rgb"]                                     # Optional if needed by tokenizer
           
#             output = model(
#                 person_lat=person_lat,garment_rgb= garment_lat,mask_lat=mask_lat,pose_lat= pose_lat,masked_person_lat=masked_person_lat
#             )
#             print(output)
#             break
    
# if __name__ == "__main__":
#     train()


from model_vton import IDM_VTON
from preprocess import VirtualTryonDataset
def main():
    # Config
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 1
    num_epochs = 100
    learning_rate = 1e-5
    dataset_path = "../train"
    print("sdfwe")
    # Initialize model
    model = IDM_VTON()
    print("sdfwe")
    model.train()
    
    # Dataset and DataLoader
    dataset = VirtualTryonDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Mixed precision
    # scaler = torch.cuda.amp.GradScaler()
    
    # Training loop
    print("SDf")
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        for i, batch in enumerate(dataloader):
            print("sdfwe")
            optimizer.zero_grad()
            
            # Mixed precision forward
            # with torch.cuda.amp.autocast():
            
            loss = model(batch)
            print("sdfwe")
            # Backpropagation
            loss.backward()
            optimizer.step()
            # scaler.update()
            
            epoch_loss += loss.item()
            print("sdfwe")
            # Log every 10 batches
            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{i}/{len(dataloader)}] "
                      f"Loss: {loss.item():.4f}")
        
        # Save checkpoint
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.4f}")
        
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f"checkpoint_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    main()