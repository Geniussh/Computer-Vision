import numpy as np
import cv2
import multiprocessing
from matplotlib import pyplot as plt
from opts import get_opts
from loadVid import loadVid
from matchPics import matchPics
from planarH import computeH_ransac, compositeH
from pathlib import Path

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


def homographyWarp(args):
    i, opts, template, cv_cover, cv_book = args
    template = cv2.resize(template, (cv_cover.shape[1], cv_cover.shape[0]))
    matches, locs1, locs2 = matchPics(cv_cover, cv_book, opts)
    bestH2to1, _ = computeH_ransac(locs1[matches[:,0]], locs2[matches[:,1]], opts)
    composite_img = compositeH(bestH2to1, template, cv_book)
    print("Frame %s Processed" % i)
    return composite_img

#Write script for Q3.1
opts = get_opts()
opts.sigma = 0.12
opts.ratio = 0.7
opts.inlier_tol = 1.4
n_worker = multiprocessing.cpu_count()

cv_cover = cv2.imread('../data/cv_cover.jpg')

# Read videos using cv2 or from previously saved .npy
ar_source = Path('ar_source.npy')
if ar_source.exists():
    ar_source = np.load('ar_source.npy')
else:
    ar_source = loadVid('../data/ar_source.mov')    # 511 frames 360x640
    np.save('ar_source.npy', ar_source)
ar_book = Path('ar_book.npy')
if ar_book.exists():
    ar_book = np.load('ar_book.npy')
else:
    ar_book = loadVid('../data/book.mov')           # 641 frames 480x640
    np.save('ar_book.npy', ar_book)

# Crop the top & bottom black bar
frame0 = ar_source[0]
x, y, w, h = cropBlackBar(frame0)
frame0 = frame0[y:y+h, x:x+w]

# Crop left and right: only the central region is used for AR
H, W = frame0.shape[:2]
width = cv_cover.shape[1] * H / cv_cover.shape[0]
wStart, wEnd = np.round([W/2 - width/2, W/2 + width/2]).astype(int)
frame0 = frame0[:, wStart:wEnd]

# Crop all frames in ar_source
new_source = np.array([f[y:y+h, x:x+w][:, wStart:wEnd] for f in ar_source])

# Multi-process
args = []
for i in range(len(new_source)):
    args.append([i, opts, new_source[i], cv_cover, ar_book[i]])
p = multiprocessing.Pool(processes=n_worker)
ar = p.map(homographyWarp, args)
p.close()
p.join()

ar = np.array(ar)
writer = cv2.VideoWriter('../result/ar.avi', cv2.VideoWriter_fourcc(*'MJPG'), 25, (ar.shape[2], ar.shape[1]))
for i, f in enumerate(ar):
    writer.write(f)
    plt.figure()
    plt.axis('off')
    plt.imshow(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    plt.savefig('../result/frame_%s.png' % i)
    plt.close()

writer.release()