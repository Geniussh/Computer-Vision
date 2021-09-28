import numpy as np
import cv2
from matchPics import matchPics
from opts import get_opts
import skimage.color
from scipy import ndimage
from helper import plotMatches
from matplotlib import pyplot as plt

opts = get_opts()
#Q2.1.6
#Read the image and convert to grayscale, if necessary
cv_cover = cv2.imread('../data/cv_cover.jpg')
angles = np.arange(0, 360, 10)
hist = np.zeros(36)

for i in range(36):
	#Rotate Image
	rotated = ndimage.rotate(cv_cover, angles[i], reshape=True)
	#Compute features, descriptors and Match features
	matches, locs1, locs2 = matchPics(cv_cover, rotated, opts)
	plotMatches(cv_cover, rotated, matches, locs1, locs2, '../result/rotation_%s.png' % i)
	#Update histogram
	hist[i] = len(matches)

#Display histogram
plt.figure()
plt.bar(angles, hist)
plt.show()
plt.savefig('../result/hist.png')