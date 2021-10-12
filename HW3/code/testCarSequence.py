import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanade import LucasKanade


parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold

seq = np.load("../data/carseq.npy")
rect = [59, 116, 145, 151]
rects = [rect]
queries = [1, 100, 200, 300, 400]
f, axs = plt.subplots(1, len(queries), dpi=1200)

for i in range(seq.shape[2]-1):
    It = seq[:,:,i]
    It1 = seq[:,:,i+1]

    p = LucasKanade(It, It1, rects[-1], threshold, num_iters)
    rect = np.array([rects[-1][:2] + p, rects[-1][2:] + p]).ravel()
    rects.append(rect)

    if i in queries:
        ax_index = queries.index(i)
        axs[ax_index].imshow(It, cmap='gray')
        box = patches.Rectangle((rect[0], rect[1]), rect[2]-rect[0], rect[3]-rect[1], ec='r', fc='none', lw=0.4)
        axs[ax_index].add_patch(box)
        axs[ax_index].set_axis_off()
        
# plt.show()
plt.savefig('../result/carseq.png', bbox_inches='tight', pad_inches=0)

with open('carseqrects.npy', 'wb') as f:
    np.save(f, np.array(rects))


# # Iterate over all rectangles
# f, ax = plt.subplots()
# for i in range(len(rects)):
#     if i % 2 == 0:
#         ax.imshow(seq[:,:,i])
#         rect = rects[i]
#         box = patches.Rectangle((rect[0], rect[1]), rect[2]-rect[0], rect[3]-rect[1], ec='r', fc='none', lw=0.4)
#         ax.add_patch(box)
#         ax.set_axis_off()
#         plt.savefig('../result/temp.png', bbox_inches='tight', pad_inches=0)