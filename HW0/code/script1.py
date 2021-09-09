from alignChannels import alignChannels
import imageio
import numpy as np
# Problem 1: Image Alignment

# 1. Load images (all 3 channels)
red = np.load("HW0/data/red.npy") 
blue = np.load("HW0/data/blue.npy") 
green = np.load("HW0/data/green.npy") 

# 2. Find best alignment
rgbResult = alignChannels(red, green, blue)

# 3. save result to rgb_output.jpg (IN THE "results" FOLDER)
imageio.imwrite('HW0/results/rgb_output.jpg', rgbResult)