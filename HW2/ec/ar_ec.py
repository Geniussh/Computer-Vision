'''
My algorithm based on ORB descriptor with FLANN based Matcher
reaches 32.8 FPS on average (never below 30 FPS) in real time 
AR application.
'''

import numpy as np
import cv2
import numpy as np
import time
import sys
from matplotlib import pyplot as plt
from loadVid import loadVid
from pathlib import Path

MIN_MATCH_COUNT = 4  # because we need at least four points to compute H

#Write script for Q4.1
def cropBlackBar(img):
    # convert to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # invert gray image
    gray = 255 - gray
    # gaussian blur
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    # threshold
    res = cv2.threshold(blur, 235, 255, cv2.THRESH_BINARY)[1]
    # use morphology to fill holes at the boundaries
    kernel = np.ones((5,5), np.uint8)
    res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel)
    res = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel)
    # invert back
    res = 255 - res
    # get contours and get bounding rectangle
    contours, _ = cv2.findContours(res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    return x, y, w, h

# using ORB instead of vanilla BRIEF
orb = cv2.ORB_create()

# using FLANN Matcher (faster but less accurate than Brute Force)
# using recommended parameters for ORB per opencv docs
# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html
FLANN_INDEX_LSH = 6
index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, 
                   key_size = 12,     
                   multi_probe_level = 1)
search_params = dict(checks = 100)  # increase checks for better precision
matcher = cv2.FlannBasedMatcher(index_params, search_params)

cv_cover = cv2.imread('../data/cv_cover.jpg')

# Read videos using cv2 or from previously saved .npy
ar_source = Path('../python/ar_source.npy')
if ar_source.exists():
    print("ar_source already exists")
    ar_source = np.load('../python/ar_source.npy')
else:
    ar_source = loadVid('../data/ar_source.mov')    # 511 frames 360x640
    np.save('../python/ar_source.npy', ar_source)
ar_book = Path('../python/ar_book.npy')
if ar_book.exists():
    print("ar_book already exists")
    ar_book = np.load('../python/ar_book.npy')
else:
    ar_book = loadVid('../data/book.mov')           # 641 frames 480x640
    np.save('../python/ar_book.npy', ar_book)

frame0 = ar_source[0]
x, y, w, h = cropBlackBar(frame0)
frame0 = frame0[y:y+h, x:x+w]
H, W = frame0.shape[:2]
width = cv_cover.shape[1] * H / cv_cover.shape[0]
wStart, wEnd = np.round([W/2 - width/2, W/2 + width/2]).astype(int)
new_source = np.array([f[y:y+h, x:x+w][:, wStart:wEnd] for f in ar_source])

cv_cover_gray = cv2.cvtColor(cv_cover, cv2.COLOR_BGR2GRAY)
kp, desc = orb.detectAndCompute(cv_cover_gray, None)

dist_threshold = 0.6  # maximum allowed relative dist btw matches
inlier_tol = 0.7
Tstart = time.time()
for i in range(len(new_source)):
    cv_book = ar_book[i]
    book_gray = cv2.cvtColor(cv_book, cv2.COLOR_BGR2GRAY)
    kp_book, desc_book = orb.detectAndCompute(book_gray, None)
    matches = matcher.knnMatch(desc, desc_book, k=2)
    filter_matches = []
    for m in matches:
        if len(m) < 2:  # bad matches
            continue
        if m[0].distance < dist_threshold * m[1].distance:
            filter_matches.append(m[0])
    locs1 = np.array([[*kp[m.queryIdx].pt] for m in filter_matches])
    locs2 = np.array([[*kp_book[m.trainIdx].pt] for m in filter_matches])

    if len(filter_matches) >= 4:
        H, _ = cv2.findHomography(locs1, locs2, cv2.RANSAC, inlier_tol)
        template = cv2.resize(new_source[i], (cv_cover.shape[1], cv_cover.shape[0]))
        mask = np.ones(template.shape)
        warped_m = cv2.warpPerspective(mask, H, (cv_book.shape[1],cv_book.shape[0]))
        warped_t = cv2.warpPerspective(template, H, (cv_book.shape[1],cv_book.shape[0]))
        composite_img = warped_t + cv_book * np.logical_not(warped_m)

        # Show Image plus FPS
        FPS = (i + 1) / (time.time() - Tstart)
        cv2.putText(composite_img, str(FPS), (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('F', composite_img)
        cv2.waitKey(1)
    else:
        print("Not enough matches are found - %d/%d" % (len(filter_matches), 4))
        break

cv2.waitKey(1)
cv2.destroyAllWindows()