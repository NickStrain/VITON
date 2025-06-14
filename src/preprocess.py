import os 
from os import listdir
from PIL import Image 
from typing import List, Tuple
import numpy as np 
from pathlib import Path

class DatasetLoader():
    def __init__(self,root_dir:str,image_size:Tuple[int,int]=(244,244),transform = None,target_image_class=None):
        """
        Args:
        """
        self.root_dir = root_dir
        self.image_size = image_size
        self.image= []
        self.image_classes = []
        self.target_image_class = target_image_class
        self.transform = transform
        # self.load_image()

        self.image_names = os.listdir(self.root_dir+"/image")
        
    def __len__(self):
        return len(self.image_names)

    # def load_image(self):
        # self.image_classes = os.listdir(self.root_dir)
        # if self.target_image_class:
        #     self.image_classes = [self.target_image_class]
        # for label_idx, class_name in enumerate(self.image_classes):
        #     class_path= os.path.join(self.root_dir,class_name)
        #     for img_path in os.listdir(class_path):
        #         self.image.append((img_path,class_name))
               

    def __getitem__(self,idx):
        # image_path, label = self.image[idx]
        person_image = np.array(Image.open(self.root_dir+"/"+"image"+"/"+self.image_names[idx]).convert("RGB"))
        garment_image = np.array(Image.open(self.root_dir+"/"+"cloth"+"/"+self.image_names[idx]).convert("RGB"))
        mask_laten =  np.array(Image.open(self.root_dir+"/"+"image-parse-agnostic-v3.2"+"/"+self.image_names[idx][:-3]+"png"))
        masked_person_lat = np.array(Image.open(self.root_dir+"/"+"agnostic-v3.2"+"/"+self.image_names[idx]))
        pose_lat = np.array(Image.open(self.root_dir+"/"+"openpose_img"+"/"+self.image_names[idx][:-4]+"_rendered.png"))
       
        if self.transform:
            person_image = self.transform(person_image)
            garment_image = self.transform(garment_image)
            mask_laten = self.transform(mask_laten)
            masked_person_lat = self.transform(masked_person_lat)
            pose_lat = self.transform(pose_lat)

        return {
            "person_lat":person_image,
            "mask_lat": mask_laten, 
            "masked_person_lat": masked_person_lat, 
            "pose_lat": pose_lat,
            "garment_rgb" : garment_image
        }
        

# data_loader = DatasetLoader("../train",)
# print(len(data_loader))

# for i in data_loader:
#     print(i)
#     break
    


