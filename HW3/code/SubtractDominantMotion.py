import numpy as np
from LucasKanadeAffine import LucasKanadeAffine
from InverseCompositionAffine import InverseCompositionAffine
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import affine_transform, generate_binary_structure, binary_erosion, binary_dilation, binary_fill_holes

def SubtractDominantMotion(image1, image2, threshold, num_iters, tolerance):
    """
    :param image1: Images at time t
    :param image2: Images at time t+1
    :param threshold: used for LucasKanadeAffine
    :param num_iters: used for LucasKanadeAffine
    :param tolerance: binary threshold of intensity difference when computing the mask
    :return: mask: [nxm]
    """
    
    mask = np.zeros(image1.shape, dtype=bool)

    # Uncomment the following line to use vanilla Lucas Kanade
    # M = LucasKanadeAffine(image1, image2, threshold, num_iters)  
    
    M = InverseCompositionAffine(image1, image2, threshold, num_iters)
    
    # # Warp It
    # Minv = np.linalg.inv(M)  # affine_transform uses inverse warping
    # warped = affine_transform(image1, Minv[:2, :2], Minv[:2, -1], image2.shape)
    h, w = image1.shape
    hrange, wrange = np.arange(h), np.arange(w)
    X, Y = np.meshgrid(wrange, hrange)
    im1 = RectBivariateSpline(hrange, wrange, image1)
    im2 = RectBivariateSpline(hrange, wrange, image2)
    warpX = M[0, 0] * X + M[0, 1] * Y + M[0, 2]
    warpY = M[1, 0] * X + M[1, 1] * Y + M[1, 2]
    outside = np.nonzero((w <= warpX) | (warpX < 0) | (h <= warpY) | (warpY < 0))
    mask2 = np.ones(image1.shape, dtype=bool)  # mask for valid coordinates
    mask2[outside] = False
    mask[np.abs(im1.ev(Y, X) - im2.ev(warpY, warpX)) > tolerance] = True
    mask &= mask2
    
    # For ant sequence
    # # Join
    # struct = generate_binary_structure(2, 1)
    # mask = binary_dilation(mask, struct, iterations=3)
    # mask = binary_erosion(mask, struct, iterations=3)
    
    # # De-noise
    # mask = binary_erosion(mask, iterations=1)
    # mask = binary_dilation(mask, iterations=1)

    # For aerial sequence
    # Join
    struct = np.array(([0,0,1,0,0],[0,1,1,1,0],[1,1,1,1,1],[0,1,1,1,0],[0,0,1,0,0]))
    mask = binary_dilation(mask, struct, iterations=3)
    mask = binary_erosion(mask, struct, iterations=3)

    return mask
