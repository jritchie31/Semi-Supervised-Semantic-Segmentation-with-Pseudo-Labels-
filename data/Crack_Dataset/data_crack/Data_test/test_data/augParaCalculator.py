import torch
import numpy as np
from PIL import Image
import os
import os.path as osp

# Get the absolute path of the current file
current_dir = osp.dirname(osp.abspath(__file__))
folder_path = osp.join(current_dir, r"labeled_Range_crop") # Replace with the path to your folder
extension = '.png' # Replace with the file extension of your images

# list all files in the folder with the specified extension
image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(extension)]

# load all the grayscale images into a list
images = [np.array(Image.open(fname).convert('L')) for fname in image_files]

# convert the list of images to a tensor
tensor = torch.stack([torch.from_numpy(img) for img in images])

# calculate the mean and std of the tensor
mean = torch.mean(tensor.float())
std = torch.std(tensor.float())

print('Mean:', mean.item())
print('Std:', std.item())
