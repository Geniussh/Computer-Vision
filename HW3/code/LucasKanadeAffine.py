import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanadeAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array] put your implementation here
    """

    M = np.eye(3)
    h, w = It.shape

    It_h, It_w = np.arange(h), np.arange(w)
    temp = RectBivariateSpline(It_w, It_h, It.T)
    img = RectBivariateSpline(It_w, It_h, It1.T)

    X, Y = [a.ravel() for a in np.meshgrid(It_w, It_h)]
    homoXY = np.vstack((X, Y, np.ones(h*w)))
    for _ in range(int(num_iters)):
        # Warp I with W
        warpedXY = M @ homoXY
        common = np.nonzero((0 <= warpedXY[0]) & (warpedXY[0] < w) &
                            (0 <= warpedXY[1]) & (warpedXY[1] < h))
        x,  y  = X[common], Y[common]  # common pixels in template
        xw, yw = warpedXY[:2, common]  # common pixels in I warped

        # Compute error
        err = temp.ev(x, y) - img.ev(xw, yw)

        # Warp gradient of I to compute (dIx, dIy)
        dIx = img.ev(xw, yw, dx=1).ravel()
        dIy = img.ev(xw, yw, dy=1).ravel()

        # Jacobian = [x y 1 0 0 0
        #             0 0 0 x y 1]
        # A = grad * Jac = [dIx*x, dIx*y, dIx, dIy*x, dIy*y, dIy]
        A = np.array([dIx * x, dIy * y, dIx, dIy * x, dIy * y, dIy]).T
        dM = np.linalg.pinv(A.T @ A) @ A.T @ err.reshape(-1, 1)

        M[:2, :] = M[:2, :] + dM.reshape(2, 3)

        if np.sum(dM**2) < threshold:
            break
    
    return M

# For debug
# seq = np.load('../data/aerialseq.npy')
# M = LucasKanadeAffine(seq[0], seq[1], 1e-2, 1e3)
# print(M)