import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

for img in os.listdir('../images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
    bboxes, bw = findLetters(im1)

    plt.imshow(bw, cmap='gray')
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    plt.show()
    
    ##########################
    #####   Find lines   #####
    ##########################
    # Find the centroids of each box
    centroids = [[b, ((b[0] + b[2]) // 2, (b[1] + b[3]) // 2)] for b in bboxes]
    # sort top-down first, and then sort left-right
    centroids = sorted(centroids, key=lambda x: (x[1][0], x[1][1]))
    lines = []
    for box, centroid in centroids:
        flag = False
        # Try to fit the box in all the lines found so far
        for line in lines:
            # Group the box in this line if its centroid's y is within
            # average y coord +/- average height of all boxes in this line
            nBoxes = len(line)
            y = sum([(b[0][0] + b[0][2]) // 2 for b in line]) / nBoxes
            h = sum([b[0][2] - b[0][0] for b in line]) / nBoxes
            if y - h < centroid[0] < y + h:
                line.append((box, centroid[1]))  # store x coordinate to sort line-wise
                flag = True
                break
        if not flag:
            lines.append([(box, centroid[1])])  # start a new line
    lines = [sorted(line, key=lambda x: x[1]) for line in lines]
    lines = [[a[0] for a in line] for line in lines]  # keep the boxes only

    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    X = []
    for line in lines:
        x = []  # cropped images per line as input to the nn
        for box in line:
            minr, minc, maxr, maxc = box
            im = bw[minr:maxr+1, minc:maxc+1]
            H, W = im.shape
            pad = max(H, W) // 6
            extra = abs((H - W) // 2)
            if H > W:
                im = np.pad(im, ((pad,)*2, (pad + extra,)*2), "constant", constant_values = (1, 1))
            else:
                im = np.pad(im, ((pad + extra,)*2, (pad,)*2), "constant", constant_values = (1, 1))
            
            # Dilate to match training images' thickness
            numErosion = 20 if '04' in img else 5
            for _ in range(numErosion):
                im = skimage.morphology.erosion(im)

            im = skimage.transform.resize(im, (32, 32))
            x.append(im.T.flatten())
        X.append(np.array(x))  # change to np array required as the input to the nn

    
    # Visualize cropped images
    # rc = {"axes.spines.left" : False,
    #     "axes.spines.right" : False,
    #     "axes.spines.bottom" : False,
    #     "axes.spines.top" : False,
    #     "xtick.bottom" : False,
    #     "xtick.labelbottom" : False,
    #     "ytick.labelleft" : False,
    #     "ytick.left" : False}
    # plt.rcParams.update(rc)
    # maxLine = max([len(x) for x in X])
    # fig, ax = plt.subplots(len(X), maxLine, figsize=(20,20), sharex=True, sharey=True)
    # plt.setp(ax.flat, aspect=1.0, adjustable='box')
    # for i, x in enumerate(X):
    #     for j, char in enumerate(x):
    #         ax[i, j].imshow(X[i][j].reshape(32,32).T, cmap='gray')
    # plt.show()

    # load the weights
    # run the crops through your neural network and print them out
    import pickle
    import string
    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    params = pickle.load(open('q3_weights.pickle','rb'))
    print(f"Text Extraction - {img} ")
    for x in X:
        h1 = forward(x, params, 'layer1')
        probs = forward(h1, params, 'output', softmax)
        preds = np.argmax(probs, axis=1)
        line = [letters[p] for p in preds]
        line = ' '.join(line)
        print(line)
    print()