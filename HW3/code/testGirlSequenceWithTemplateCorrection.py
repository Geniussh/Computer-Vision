import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanade import LucasKanade


parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--template_threshold', type=float, default=5, help='threshold for determining whether to update template')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
template_threshold = args.template_threshold
_rects = np.load('girlseqrects.npy')  # rect for naive template update

seq = np.load("../data/girlseq.npy")
rect = [280, 152, 330, 318]
rects = [rect]
queries = [1, 20, 40, 60, 80]
f, axs = plt.subplots(1, len(queries), dpi=1200)
T = seq[:,:,0]   # template to be updated
T0 = seq[:,:,0]  # the first template

for i in range(seq.shape[2]-1):
    # print("Processing Frame %s" % i)
    p = LucasKanade(T, seq[:,:,i+1], rects[-1], threshold, num_iters) # Eq 4 in Matthews
    
    pn = np.array([rects[-1][0] - rects[0][0] + p[0], rects[-1][1] - rects[0][1] + p[1]])
    pstar = LucasKanade(T0, seq[:,:,i+1], rects[0], threshold, num_iters, pn) # Eq 5 in Matthews
    
    if np.linalg.norm(pstar - pn) <= template_threshold:
        # Naive Update
        T = seq[:,:,i+1]
        pstar = np.array([rects[0][0] - rects[-1][0] + pstar[0], rects[0][1] - rects[-1][1] + pstar[1]])
        rect = np.array([rects[-1][:2] + pstar, rects[-1][2:] + pstar]).ravel()
    else:
        # Use the same template as previous iteration
        rect = np.array([rects[-1][:2] + p, rects[-1][2:] + p]).ravel()
    
    rects.append(rect)

    if i in queries:
        ax_index = queries.index(i)
        axs[ax_index].imshow(seq[:,:,i], cmap='gray')
        box = patches.Rectangle((rect[0], rect[1]), rect[2]-rect[0], rect[3]-rect[1], ec='r', fc='none', lw=0.4)
        _rect = _rects[i+1]
        _box = patches.Rectangle((_rect[0], _rect[1]), _rect[2]-_rect[0], _rect[3]-_rect[1], ec='b', fc='none', lw=0.4)
        axs[ax_index].add_patch(box)
        axs[ax_index].add_patch(_box)
        axs[ax_index].set_axis_off()
        
# plt.show()
plt.savefig('../result/girlseq_wcrt.png', bbox_inches='tight', pad_inches=0)

with open('girlseqrects-wcrt.npy', 'wb') as f:
    np.save(f, np.array(rects))
