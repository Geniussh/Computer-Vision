import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FFMpegWriter
from SubtractDominantMotion import SubtractDominantMotion

frames = np.load('../data/carseq.npy')
rects_r = np.load('carseqrects-wcrt.npy')
rects_b = np.load('carseqrects.npy')

writer = FFMpegWriter(fps=15)

# fig = plt.figure()
# ax = fig.gca()
# with writer.saving(fig, "car.mp4", 500):
#     for i in range(frames.shape[2]):
#         ax.imshow(frames[:,:,i], cmap='gray')
#         rect = rects_r[i]
#         box = patches.Rectangle((rect[0], rect[1]), rect[2]-rect[0], rect[3]-rect[1], ec='r', fc='none', lw=0.4)
#         _rect = rects_b[i]
#         _box = patches.Rectangle((_rect[0], _rect[1]), _rect[2]-_rect[0], _rect[3]-_rect[1], ec='b', fc='none', lw=0.4)
#         ax.add_patch(box)
#         ax.add_patch(_box)
#         ax.set_axis_off()
#         fig.tight_layout(pad=0)
#         ax.margins(0)
#         writer.grab_frame()
#         plt.cla()
    

seq = np.load('../data/aerialseq.npy')
fig = plt.figure()
ax = fig.gca()

with writer.saving(fig, "aerial.mp4", 500):
    for i in range(seq.shape[2]-1):
        It = seq[:,:,i]
        It1 = seq[:,:,i+1]
        mask = SubtractDominantMotion(It, It1, 1e-3, 1e3, 0.14)
        ax.imshow(It, 'gray', interpolation='none')
        ax.imshow(np.ma.masked_where(mask == 0, mask), 'cool', interpolation='none', alpha=0.6)
        ax.set_axis_off()
        fig.tight_layout(pad=0)
        ax.margins(0)
        writer.grab_frame()
        plt.cla()