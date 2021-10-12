import numpy as np
from scipy.interpolate import RectBivariateSpline

def InverseCompositionAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array]
    """

    # put your implementation here
    M = np.eye(3)
    h, w = It.shape

    It_h, It_w = np.arange(h), np.arange(w)
    temp = RectBivariateSpline(It_w, It_h, It.T)
    img = RectBivariateSpline(It_w, It_h, It1.T)

    X, Y = [a.ravel() for a in np.meshgrid(It_w, It_h)]
    homoXY = np.vstack((X, Y, np.ones(h*w)))
    
    # dT
    dTx = temp.ev(X, Y, dx=1).ravel()
    dTy = temp.ev(X, Y, dy=1).ravel()

    # dWdp at p=0 is [X Y 1 0 0 0
    #                 0 0 0 X Y 1]
    # A = grad * dWdp = [dTx*X, dTx*Y, dTx, dTy*X, dTy*Y, dTy]
    # A_dagger is A's pesudo-inverse, (A.T A)^-1 A.T
    A = np.array([dTx * X, dTx * Y, dTx, dTy * X, dTy * Y, dTy]).T
    A_dagger = np.linalg.pinv(A.T @ A) @ A.T

    for _ in range(int(num_iters)):
        # Warp I with W
        warpedXY = M @ homoXY
        outOfBound = np.nonzero((w <= warpedXY[0]) | (warpedXY[0] < 0) |
                                (h <= warpedXY[1]) | (warpedXY[1] < 0))
        
        # Compute error
        # Note here I calculated I(W(x;p)) - T(x) instead of 
        # the other way around because I include the negative sign
        # of dp (in the dp closed form equation) inside the error
        err = img.ev(warpedXY[0], warpedXY[1]) - temp.ev(X, Y)
        err[outOfBound] = 0  # set error to 0 for x outside img

        # Compute dp
        dp = A_dagger @ err.reshape(-1,1)

        # Update W(x;p) <-- W(x;p) @ W(x;dp)^-1
        dM = np.array([[1+dp[0],   dp[1], dp[2]], 
                       [  dp[3], 1+dp[4], dp[5]],
                       [      0,       0,     1]], dtype=np.float)
        M = M @ np.linalg.inv(dM)

        if np.sum(dp**2) < threshold:
            break
    
    return M

# For debug
# seq = np.load('../data/aerialseq.npy')
# M = InverseCompositionAffine(seq[0], seq[1], 1e-2, 1e3)
# print(M)