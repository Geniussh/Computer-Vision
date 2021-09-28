import numpy as np
import cv2
import sys
sys.path.append('../python')
from matplotlib import pyplot as plt
from matchPics import matchPics
from helper import plotMatches
from planarH import computeH_ransac, compositeH
from opts import get_opts

#Write script for Q4.2x
opts = get_opts()
opts.sigma = 0.12
opts.ratio = 0.7
opts.inlier_tol = 1.4

# left = cv2.imread('../data/pano_left.jpg')
# right = cv2.imread('../data/pano_right.jpg')
left = cv2.imread('../data/DJI_0001.JPG')
right = cv2.imread('../data/DJI_0002.JPG')

matches, locs1, locs2 = matchPics(left, right, opts)
H, _ = computeH_ransac(locs1[matches[:,0]], locs2[matches[:,1]], opts)

hl, wl = left.shape[:2]
hr, wr = right.shape[:2]
res = cv2.warpPerspective(right, H, (wr + wl, hr + hl))
res[:hl, :wl] = left

# Cropping the black area
# convert to grayscale
gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
# threshold
gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
# get contours and get bounding rectangle
contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))

# plt.figure()
# plt.axis('off')
# plt.imshow(cv2.cvtColor(res[y:y+h, x:x+w], cv2.COLOR_BGR2RGB))
# plt.savefig('../result/panaroma_example.jpg')

cv2.imwrite('../result/panaroma.jpg', res[y:y+h, x:x+w])
# cv2.imwrite('../result/panaroma_example.jpg', res[y:y+h, x:x+w])