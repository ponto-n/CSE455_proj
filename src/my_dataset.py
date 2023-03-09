import torch
import random
from math import pi
from torch.utils.data import Dataset
import torch
from torch.utils.data import Dataset

class DebugRotationDataset(Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return len(self.size)

    def __getitem__(self, idx):
      return {"pixel_values": torch.rand(224, 224, 3), "labels": torch.Tensor(random.uniform(0, 2*pi))}
  