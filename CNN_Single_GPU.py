import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import torchvision.datasets as datasets
import torchvision.transforms as transforms
# Mixed precision imports
from sklearn.model_selection import train_test_split

import kagglehub

# Download latest version
path = kagglehub.dataset_download("chethuhn/water-bottle-dataset")

print("Path to dataset files:", path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

