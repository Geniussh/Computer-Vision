from matplotlib.pyplot import connect
import numpy as np

import skimage
from skimage.measure import label, regionprops
from skimage.color import rgb2gray
from skimage.restoration import estimate_sigma, denoise_bilateral
from skimage.filters import threshold_otsu
from skimage.morphology import closing, square
from skimage.segmentation import clear_border

# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes = []
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes 
    # this can be 10 to 15 lines of code using skimage functions

    # Estimate noise & Denoise
    sigma = estimate_sigma(image, average_sigmas=True, multichannel=True)
    image = denoise_bilateral(image, sigma_color=sigma, multichannel=True)

    # greyscale
    image = rgb2gray(image)

    # threshold -> morphology -> label
    thresh = threshold_otsu(image)
    bw = closing(image < thresh, square(5))
    cleared = clear_border(bw)
    label_image = label(bw, connectivity=2)

    # skip small boxes
    regions = regionprops(label_image)
    area_thresh = sum([region.area for region in regions]) / len(regions) / 3
    for region in regions:
        if region.area >= area_thresh:
            bboxes.append(region.bbox)

    return bboxes, (~bw).astype(float)