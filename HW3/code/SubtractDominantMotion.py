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

    # Uncomment the following line to use vanilla Lucas Kanade
    # M = LucasKanadeAffine(image1, image2, threshold, num_iters)  
    
    M = InverseCompositionAffine(image1, image2, threshold, num_iters)

    mask = np.ones(image1.shape, dtype=bool)
    warped = affine_transform(image1, M)
    warped = binary_erosion(warped)
    warped = binary_dilation(warped)
    warped = binary_dilation(warped)
    warped = binary_erosion(warped)
    mask = np.abs(image1 - warped) > tolerance
    
    # Join
    # struct = generate_binary_structure(2, 1)
    # mask = binary_dilation(mask, struct, iterations=3)
    # mask = binary_erosion(mask, struct, iterations=3)
    
    # # De-noise
    # mask = binary_erosion(mask, iterations=1)
    # mask = binary_dilation(mask, iterations=1)

    # Join
    # struct = np.array(([0,0,1,0,0],[0,1,1,1,0],[1,1,1,1,1],[0,1,1,1,0],[0,0,1,0,0]))
    # mask = binary_dilation(mask, struct, iterations=3)
    # mask = binary_erosion(mask, struct, iterations=3)

    return mask
