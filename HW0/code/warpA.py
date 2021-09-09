import numpy as np

def warp(im, A, output_shape):
    """ Warps (h,w) image im using affine (3,3) matrix A
    producing (output_shape[0], output_shape[1]) output image
    with warped = A*input, where warped spans 1...output_size.
    Uses nearest neighbor interpolation."""
    
    height, width = output_shape[0], output_shape[1]
    coords = np.flipud(np.indices((width, height)).reshape(2, -1))  # flipud to make y coords over x coords to follow scipy traditions
    coords = np.vstack((coords, np.ones(coords.shape[1]))).astype(int)  # homogenenous coords in dest
    xd, yd = coords[1], coords[0]  # coords in dest

    invWarp =  np.round(np.linalg.inv(A) @ coords).astype(int)
    xs, ys = invWarp[1], invWarp[0]  # coords in src via inverse warping

    # Get pixels within image boundaries
    indices = np.where((xs >= 0) & (xs < im.shape[1]) &
                        (ys >= 0) & (ys < im.shape[0]))

    output = np.zeros_like(im)
    output[yd[indices], xd[indices]] = im[ys[indices], xs[indices]]

    return output
