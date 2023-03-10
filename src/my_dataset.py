import torch
import random
from PIL import Image
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
