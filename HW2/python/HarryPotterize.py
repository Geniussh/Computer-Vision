import numpy as np
import cv2
import skimage.io 
import skimage.color
from opts import get_opts
from matplotlib import pyplot as plt

#Import necessary functions
from matchPics import matchPics
from planarH import computeH_ransac
from planarH import compositeH
from helper import plotMatches

#Write script for Q2.2.4
opts = get_opts()

cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png')
hp_cover = cv2.imread('../data/hp_cover.jpg')

# Reshape hp_cover so that it would fill up the same space as the book
hp_cover = cv2.resize(hp_cover, (cv_cover.shape[1], cv_cover.shape[0]))

matches, locs1, locs2 = matchPics(cv_cover, cv_desk, opts)
plotMatches(cv_cover, cv_desk, matches, locs1, locs2, 'match.png')

bestH2to1, inliers = computeH_ransac(locs1[matches[:,0]], locs2[matches[:,1]], opts)

composite_img = compositeH(bestH2to1, hp_cover, cv_desk)

plt.figure()
plt.axis('off')
plt.imshow(cv2.cvtColor(composite_img, cv2.COLOR_BGR2RGB))
plt.savefig('../result/composite.png')

# # Hyperparameter Tuning
# for max_iter in [10, 50, 100, 500, 1000, 2000]:
#     opts.max_iters = max_iter
#     matches, locs1, locs2 = matchPics(cv_cover, cv_desk, opts)
#     bestH2to1, inliers = computeH_ransac(locs1[matches[:,0]], locs2[matches[:,1]], opts)
#     composite_img = compositeH(bestH2to1, hp_cover, cv_desk)

#     plt.figure()
#     plt.axis('off')
#     plt.imshow(cv2.cvtColor(composite_img, cv2.COLOR_BGR2RGB))
#     plt.savefig('../result/composite_%s_%s.png' % (opts.max_iters, opts.inlier_tol))

# opts.max_iters = 500
# for tol in [0.125, 0.25, 0.5, 1.0, 2.0, 4.0]:
#     opts.inlier_tol = tol
#     matches, locs1, locs2 = matchPics(cv_cover, cv_desk, opts)
#     bestH2to1, inliers = computeH_ransac(locs1[matches[:,0]], locs2[matches[:,1]], opts)
#     composite_img = compositeH(bestH2to1, hp_cover, cv_desk)

#     plt.figure()
#     plt.axis('off')
#     plt.imshow(cv2.cvtColor(composite_img, cv2.COLOR_BGR2RGB))
#     plt.savefig('../result/composite_%s_%s.png' % (opts.max_iters, opts.inlier_tol))