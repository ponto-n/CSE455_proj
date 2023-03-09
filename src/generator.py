import torch
import random
from math import pi
from torch.utils.data import Dataset
import torch
from torch.utils.data import Dataset

class RotationDataset(Dataset):
    def __init__(self, quantity):
        self.quantity = quantity

    def __len__(self):
        return len(self.quantity)

    def __getitem__(self, idx):
      res = dict()
      pixels = torch.rand(224, 224, 3)
      res[pixels] = torch.Tensor(random.uniform(0, 2*pi))
      
      return res
  