import torch
import random


from torch.utils.data import Dataset
import torch

import numpy as np

class DebugRotationDataset(Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
      return {"pixel_values": torch.randn(3, 224, 224), "labels": torch.rand([]) * 2 * np.pi}
  