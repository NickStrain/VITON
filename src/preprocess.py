import os 
from os import listdir
from PIL import Image 
from typing import List, Tuple
import numpy as np 
from pathlib import Path

class DatasetLoader():
    def __init__(self,root_dir:str,image_size:Tuple[int,int]=(244,244),transform = None):
        """
        Args:
        """
        self.root_dir = root_dir
        self.image_size = image_size
        self.image= []
        self.image_classes = []
        self.transform = transform
        self.load_image()

    def __len__(self):
        return len(self.image)

    def load_image(self):
        self.image_classes = os.listdir(self.root_dir)
        for label_idx, class_name in enumerate(self.image_classes):
            class_path= os.path.join(self.root_dir,class_name)
            for img_path in os.listdir(class_path):
                self.image.append((img_path,class_name))
               

    def __getitem__(self,idx):
        image_path, label = self.image[idx]
        image = Image.open(self.root_dir+"/"+label+"/"+image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
        

data_loader = DatasetLoader("../train")

for i,a in data_loader:
    print(i,a)
    


