import numpy as np

def alignChannels(red, green, blue):
    """Given 3 images corresponding to different channels of a color image,
    compute the best aligned result with minimum abberations

    Args:
      red, green, blue - each is a HxW matrix corresponding to an HxW image

    Returns:
      rgb_output - HxWx3 color image output, aligned as desired"""

    H, W = red.shape
    original = np.zeros((H + 60, W + 60))
    original[30:H+30, 30:W+30] = red
    green_norm = np.sqrt(np.sum(green * green))
    maxScore = -1
    bestij = None
    for i in range(61):
        for j in range(61):
            batch = original[i:H+i, j:W+j]
            score = np.sum(green * batch) / \
                (green_norm * np.sqrt(np.sum(batch * batch)))
            if score > maxScore:
                maxScore = score
                bestij = (i,j)
    _green = np.zeros((H + 60, W + 60))
    _green[bestij[0]:H+bestij[0], bestij[1]:W+bestij[1]] = green
    
    blue_norm = np.sqrt(np.sum(blue * blue))
    maxScore = -1
    for i in range(61):
        for j in range(61):
            batch = original[i:H+i, j:W+j]
            score = np.sum(blue * batch) / \
                (blue_norm * np.sqrt(np.sum(batch * batch)))
            if score > maxScore:
                maxScore = score
                bestij = (i,j)
    _blue = np.zeros((H + 60, W + 60))
    _blue[bestij[0]:H+bestij[0], bestij[1]:W+bestij[1]] = blue
    
    return np.stack((original, _green, _blue), axis=2).astype(int)