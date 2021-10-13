import argparse
import numpy as np
import matplotlib.pyplot as plt
from SubtractDominantMotion import SubtractDominantMotion
import time

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e3, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--tolerance', type=float, default=0.025, help='binary threshold of intensity difference when computing the mask')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
tolerance = args.tolerance

seq = np.load('../data/antseq.npy')
queries = [30, 60, 90, 120]
f, axs = plt.subplots(1, len(queries), dpi=1200)

start = time.time()
for i in range(seq.shape[2]-1):
    It = seq[:,:,i]
    It1 = seq[:,:,i+1]

    mask = SubtractDominantMotion(It, It1, threshold, num_iters, tolerance)

    if i in queries:
        ax_index = queries.index(i)
        axs[ax_index].imshow(It, 'gray', interpolation='none')
        axs[ax_index].imshow(np.ma.masked_where(mask == 0, mask), 'cool', interpolation='none', alpha=0.6)
        axs[ax_index].set_axis_off()

print("Time used: %s" % (time.time() - start))
plt.savefig('../result/ant.png', bbox_inches='tight', pad_inches=0)