import os 
from os import listdir
from PIL import Image 
from typing import List, Tuple
import numpy as np 
from pathlib import Path


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from diffusers import StableDiffusionXLInpaintPipeline, DDPMScheduler
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
import numpy as np
from PIL import Image
from torchvision import transforms

# class DatasetLoader():
#     def __init__(self,root_dir:str,image_size:Tuple[int,int]=(244,244),transform = None,target_image_class=None):
#         """
#         Args:
#         """
#         self.root_dir = root_dir
#         self.image_size = image_size
#         self.image= []
#         self.image_classes = []
#         self.target_image_class = target_image_class
#         self.transform = transform
#         # self.load_image()

#         self.image_names = os.listdir(self.root_dir+"/image")
        
#     def __len__(self):
#         return len(self.image_names)

#     # def load_image(self):
#         # self.image_classes = os.listdir(self.root_dir)
#         # if self.target_image_class:
#         #     self.image_classes = [self.target_image_class]
#         # for label_idx, class_name in enumerate(self.image_classes):
#         #     class_path= os.path.join(self.root_dir,class_name)
#         #     for img_path in os.listdir(class_path):
#         #         self.image.append((img_path,class_name))
               

#     def __getitem__(self,idx):
#         # image_path, label = self.image[idx]
#         person_image = Image.open(self.root_dir+"/"+"image"+"/"+self.image_names[idx]).convert("RGB")
#         garment_image = Image.open(self.root_dir+"/"+"cloth"+"/"+self.image_names[idx]).convert("RGB")
#         mask_laten =  Image.open(self.root_dir+"/"+"image-parse-agnostic-v3.2"+"/"+self.image_names[idx][:-3]+"png")
#         masked_person_lat = Image.open(self.root_dir+"/"+"agnostic-v3.2"+"/"+self.image_names[idx])
#         pose_lat = Image.open(self.root_dir+"/"+"openpose_img"+"/"+self.image_names[idx][:-4]+"_rendered.png")
       
#         if self.transform:
#             person_image = self.transform(person_image)
#             garment_image = self.transform(garment_image)
#             mask_laten = self.transform(mask_laten)
#             masked_person_lat = self.transform(masked_person_lat)
#             pose_lat = self.transform(pose_lat)

#         return {
#             "person_lat":person_image,
#             "mask_lat": mask_laten, 
#             "masked_person_lat": masked_person_lat, 
#             "pose_lat": pose_lat,
#             "garment_rgb" : garment_image
#         }
        

# # data_loader = DatasetLoader("../train",)
# # print(len(data_loader))

# # for i in data_loader:
# #     print(np.array(i["mask_lat"]).shape)
# #     break
    


class VirtualTryonDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = self._load_samples()
        
    def _load_samples(self):
        # Should return list of dicts with:
        # {'person_img': path, 'garment_img': path, 
        #  'mask': path, 'pose': path, 'caption': text}
        return [...]  # Implement based on your dataset
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        person_image = Image.open(self.root_dir+"/"+"image"+"/"+self.image_names[idx]).convert("RGB")
        garment_image = Image.open(self.root_dir+"/"+"cloth"+"/"+self.image_names[idx]).convert("RGB")
        mask_laten =  Image.open(self.root_dir+"/"+"image-parse-agnostic-v3.2"+"/"+self.image_names[idx][:-3]+"png")
        masked_person_lat = Image.open(self.root_dir+"/"+"agnostic-v3.2"+"/"+self.image_names[idx])
        pose_lat = Image.open(self.root_dir+"/"+"openpose_img"+"/"+self.image_names[idx][:-4]+"_rendered.png")
        
        # Load images
        # person_img = Image.open(sample['person_img']).convert('RGB')
        # garment_img = Image.open(sample['garment_img']).convert('RGB')
        # mask = Image.open(sample['mask']).convert('L')
        # pose = Image.open(sample['pose']).convert('RGB')
        
        # Apply transforms
        if self.transform:
            person_img = self.transform(person_image)
            garment_img = self.transform(garment_image)
            mask = self.transform(mask_laten)
            pose = self.transform(pose_lat)
        else:
            # Default resize and normalize
            transform = transforms.Compose([
                transforms.Resize((512, 384)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
            person_img = transform(person_img)
            garment_img = transform(garment_img)
            pose = transform(pose)
            mask = transforms.Resize((64, 64))(mask)  # Latent space size
            mask = transforms.ToTensor()(mask)
        
        return {
            'person_img': person_img,
            'garment_img': garment_img,
            'mask': mask,
            'pose': pose,
            'caption': sample['caption']
        }

