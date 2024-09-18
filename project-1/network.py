import os
import numpy as np
import glob
import PIL.Image as Image
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        #...
    def forward(self, x):
        #...
        return x