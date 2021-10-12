from os import wait
import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanade(It, It1, rect, threshold, num_iters, p0=np.zeros(2)):
    """
    :param It: template image
    :param It1: Current image
    :param rect: Current position of the car (top left, bot right coordinates)
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :param p0: Initial movement vector [dp_x0, dp_y0]
    :return: p: movement vector [dp_x, dp_y]
    """
	
    # Put your implementation here
    p = p0

    w, h = np.round(rect[2] - rect[0] + 1).astype(int), np.round(rect[3] - rect[1] + 1).astype(int)
    Y, X = np.meshgrid(np.linspace(start=rect[1], stop=rect[3], num=h, endpoint=True),
                       np.linspace(start=rect[0], stop=rect[2], num=w, endpoint=True))
    
    It_h, It_w = np.arange(It.shape[0]), np.arange(It.shape[1])
    temp = RectBivariateSpline(It_h, It_w, It)
    T = temp.ev(Y, X)
    img = RectBivariateSpline(It_h, It_w, It1)

    grad = np.zeros((w*h, 2))
    for _ in range(int(num_iters)):
        # Warp I with W
        IWp = img.ev(Y + p[1], X + p[0])

        # Compute error image
        err = T - IWp

        # Warp gradient of I to compute (dIx, dIy)
        grad[:, 0] = img.ev(Y + p[1], X + p[0], dy = 1).reshape(1,-1)
        grad[:, 1] = img.ev(Y + p[1], X + p[0], dx = 1).reshape(1,-1)

        # Evaluate Jacobian
        Jac = np.eye(2)  # for pure translational p
        
        # Compute Hessian
        A = grad @ Jac
        H = A.T @ A
        dp = np.linalg.inv(H) @ A.T @ err.reshape(-1, 1)

        # Update parameters p <-- p + dp
        p = p + dp.ravel()
        
        if (np.linalg.norm(dp))**2 < threshold:
            break

    return p
