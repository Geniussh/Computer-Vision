import numpy as np
import cv2
from matchPics import matchPics
from helper import plotMatches
from opts import get_opts

opts = get_opts()

cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png')

for sigma in [0.09, 0.12, 0.15, 0.18, 0.21]:
    opts.sigma = sigma
    matches, locs1, locs2 = matchPics(cv_cover, cv_desk, opts)

    #display matched features
    plotMatches(cv_cover, cv_desk, matches, locs1, locs2, '../result/match_%s_%s.png' % (opts.sigma, opts.ratio))

for ratio in [0.5, 0.6, 0.7, 0.8, 0.9]:
    opts.ratio = ratio
    matches, locs1, locs2 = matchPics(cv_cover, cv_desk, opts)

    #display matched features
    plotMatches(cv_cover, cv_desk, matches, locs1, locs2, '../result/match_%s_%s.png' % (opts.sigma, opts.ratio))