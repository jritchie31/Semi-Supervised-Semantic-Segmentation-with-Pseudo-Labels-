import torch
from PIL import Image
import os.path as osp
import numpy as np

# Get the absolute path of the current file
current_dir = osp.dirname(osp.abspath(__file__))
dir = osp.join(current_dir, r"labeled_Segmentation_crop/LcmsResult_ImageRng_000145_0.png")
img = Image.open(dir)

# Convert image to tensor
img_tensor = torch.Tensor(np.array(img))

# Normalize tensor
img_tensor_max = img_tensor.max()
img_tensor_min = img_tensor.min()
img_tensor_size = img_tensor.size() # change tensor shape from HxWxC to CxHxW
img_tensor = img_tensor / 255.0 # normalize to [0, 1]
img_tensor = (img_tensor - 0.5) / 0.5 # normalize to [-1, 1]
