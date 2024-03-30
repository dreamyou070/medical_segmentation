# data agumenting
# Compare this snippet from data_augmenting.py:
import os

# [1] rotation
import numpy as np
from PIL import Image
from torchvision import transforms
transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize([0.5], [0.5]), ])
"""
input = np.random.rand(3,3,3)
output = np.rot90(input, k=4)
print(f'input = {input}')
print(f'output = {output}')
"""
img_dir = 'brats20_training_001_28.jpg'
input_array = np.array(Image.open(img_dir).convert('RGB').resize((512,512)))
output_array = np.rot90(input_array, k=2)
output = transform(output_array.copy())
print(f'input_array = {input_array.shape}')
print(f'output_array = {output_array.shape}')