# data agumenting
# Compare this snippet from data_augmenting.py:
import os

# [1] rotation
import numpy as np



input = np.random.rand(3,3,3)
output = np.rot90(input, k=4)
print(f'input = {input}')
print(f'output = {output}')