import os
import pathlib

import torch
import random
from PIL import Image
import imghdr
import preprocessor
from preprocessor import create_connection
from torchvision import transforms

from torch.utils.data import Dataset
import torch

import numpy as np

ROTATED_FOLDER_DIR = r"data/rotated_birds"


class DebugRotationDataset(Dataset):
    def __init__(self, size, label):
        self.size = size
        self.label = label
        self.dict = {}
        self.read_db()

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            "pixel_values": self.dict[idx][0],
            "labels": self.dict[idx][1],
        }

    def read_db(self):
        conn = create_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT * FROM birds_split where label = (?)", [self.label])
            for idx, entry in enumerate(cur.fetchmany(self.size)):
                filepath = rf"{ROTATED_FOLDER_DIR}/{entry[0]}"
                img = Image.open(filepath)
                transform = transforms.Compose([transforms.ToTensor()])
                tensor = transform(img)
                self.dict[idx] = (tensor, entry[1])
        except Exception as e:
            print(e)

class RotationDataset(Dataset):
    def __init__(self, image_dir, transform, max_size=None):
        super().__init__()
        self.files = [file for file in pathlib.Path(image_dir).rglob("*.[jJ][pP][gG]")]
        if max_size is not None:
            self.files = self.files[:max_size]
        self.transform = transform
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        og_width = img.width
        og_height = img.height

        radians_rotated = random.random() * 2 * np.pi
        img = img.rotate(radians_rotated / (2 * np.pi) * 360)

        largest_rect = preprocessor.largest_rotated_rect(og_width, og_height, radians_rotated)

        img = preprocessor.crop_around_center(img, int(largest_rect[0]), int(largest_rect[1]))

        # Crop and resize
        img = self.transform(img)
        return {
            "pixel_values": img,
            "labels": torch.tensor(radians_rotated)
        }




        



