import numpy as np
from util import refineF
from scipy.optimize import leastsq

'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    # Replace pass by your implementation
    N = len(pts1)

    # Scale/Normalize
    pts1_ = pts1 / M
    pts2_ = pts2 / M

    A = [[a[0]*b[0], b[0]*a[1], b[0], b[1]*a[0], b[1]*a[1], 
          b[1], a[0], a[1], 1] for a, b in zip(pts1_[:N], pts2_[:N])]
    A = np.array(A)

    # Compute F from the correspondences
    w, v = np.linalg.eig(A.T @ A)
    w = np.abs(w)
    F = v[:, np.argmin(w)].reshape(3, 3)

    # Enforce the singularity condition of F
    u, s, vh = np.linalg.svd(F)
    s[-1] = 0
    F = u @ np.diag(s) @ vh

    # Refine F by using local minimization
    F = refineF(F, pts1_, pts2_)

    # Unscale/Unnormalize F
    T = np.diag([1/M, 1/M, 1])
    F = T.T @ F @ T

    return F


'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    return K2.T @ F @ K1


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
    P = []
    err = 0
    _pts1, _pts2 = pts1.astype('float'), pts2.astype('float')
    for p1, p2 in zip(_pts1, _pts2):
        # A = np.array([[C1[2,0]*p1[0] - C1[0,0], C1[2,1]*p1[0] - C1[0,1], C1[2,2]*p1[0] - C1[0,2], C1[2,3]*p1[0] - C1[0,3]],
        #               [C1[2,0]*p1[1] - C1[1,0], C1[2,1]*p1[1] - C1[1,1], C1[2,2]*p1[1] - C1[1,2], C1[2,3]*p1[1] - C1[1,3]],
        #               [C2[2,0]*p2[0] - C2[0,0], C2[2,1]*p2[0] - C2[0,1], C2[2,2]*p2[0] - C2[0,2], C2[2,3]*p2[0] - C2[0,3]],
        #               [C2[2,0]*p2[1] - C2[1,0], C2[2,1]*p2[1] - C2[1,1], C2[2,2]*p2[1] - C2[1,2], C2[2,3]*p2[1] - C2[1,3]]
        #             ])
        A1 = np.array([[0, -1,  p1[1]], 
                       [1,  0, -p1[0]]]) @ C1
        A2 = np.array([[0, -1,  p2[1]], 
                       [1,  0, -p2[0]]]) @ C2
        A = np.vstack((A1, A2))

        w, v = np.linalg.eig(A.T @ A)
        w = np.abs(w)
        Pi = v[:, np.argmin(w)]
        P.append(Pi[:3] / Pi[-1])
        
        # Reprojection
        Pi = Pi.reshape(4,1)
        p1_est = C1 @ Pi
        p1_est = (p1_est[:2] / p1_est[-1]).ravel()
        p2_est = C2 @ Pi
        p2_est = (p2_est[:2] / p2_est[-1]).ravel()
        err += np.sum((p1_est - p1)**2) + np.sum((p2_est - p2)**2)

    return np.array(P), err

'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''
def epipolarCorrespondence(im1, im2, F, x1, y1):
    def gkern(l=5, sig=1.):
        # creates a lxl gaussian kernel with sigma 'sig'
        ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
        gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
        kernel = np.outer(gauss, gauss)
        kernel = kernel / np.sum(kernel)
        return np.array([kernel]*3)

    W = 10  # half of the window size
    k = gkern(2*W)  # gaussian weighting of the window
    height, width = im1.shape[0], im1.shape[1]

    l2 = F @ np.array([x1, y1, 1]).reshape(-1,1)  # epipolar line on im2
    num = 30  # Look at 30 pixels per above/below y1, the row entry
    y2s = np.arange(y1 - num, y1 + num)
    x2s = (-l2[1] * y2s - l2[2]) / l2[0]  # Corresponding x coord on epipolar line
    
    inBound = lambda y, x: True if 0 <= y < height and 0 <= x < width else False
    w1 = im1[y1-W:y1+W, x1-W:x1+W] #* k

    # Search
    minDist = float('inf')
    x2, y2 = -1, -1
    for x,y in zip(x2s, y2s):
        x, y = round(x), round(y)
        if not inBound(y-W, x-W) or not inBound(y+W, x+W):
            continue
        w2 = im2[y-W:y+W, x-W:x+W] #* k
        dist = np.sqrt(np.sum((w1 - w2)**2))
        if dist < minDist:
            x2, y2 = x, y
            minDist = dist
    
    return x2, y2
        

'''
Q5.1: Extra Credit RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers
'''
def ransacF(pts1, pts2, M, nIters=1000, tol=3.5):
    N = len(pts1)
    bestInliers = None
    max = -1  # maximum number of inliers
    bestF = None
    for _ in range(nIters):
        indices = np.random.choice(N, 8) # pick 8 correspondences randomly
        pts8_1, pts8_2 = pts1[indices], pts2[indices]
        F = eightpoint(pts8_1, pts8_2, M)

        inliers = []
        for p1, p2 in zip(pts1, pts2):
            l2 = F @ np.array([p1[0], p1[1], 1]).reshape(-1,1)  # epipolar line on I2
            dist = (l2[0] * p2[0] + l2[1] * p2[1] + l2[2]) / np.sqrt(l2[0]**2 + l2[1]**2)
            if abs(dist) <= tol:
                inliers.append(True)
            else: inliers.append(False)
        
        # if np.sum(inliers) == len(inliers):
        #     print("All points are inliers. Lower the tolerance!")
        #     break

        if np.sum(inliers) > max:
            bestF = F
            max = np.sum(inliers)
            bestInliers = np.array(inliers)
    
    return bestF, bestInliers


'''
Q5.2:Extra Credit  Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    theta = np.linalg.norm(r)
    k = r.ravel() / theta
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])
    
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

'''
Q5.2:Extra Credit  Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    theta = np.arccos((np.trace(R) - 1) / 2)
    omega = np.array([[R[2,1] - R[1,2]],
                      [R[0,2] - R[2,0]],
                      [R[1,0] - R[0,1]]]) / (2 * np.sin(theta))
    return theta * omega

'''
Q5.3: Extra Credit Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenation of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    N = len(p1)
    r2, t2 = x[3*N:3*N+3], x[3*N+3:]
    P = np.hstack((x[:3*N].reshape(-1,3), np.ones((N,1))))  # Nx4
    M2 = np.hstack((rodrigues(r2), t2.reshape(-1,1)))
    C2 = K2 @ M2
    C1 = K1 @ M1

    p1_hat = C1 @ P.T  # 3xN
    p1_hat = np.divide(p1_hat, p1_hat[-1])  # scale to homogeneous
    p1_hat = p1_hat[:2].T  # Nx2
    
    p2_hat = C2 @ P.T
    p2_hat = np.divide(p2_hat, p2_hat[-1])  # scale to homogeneous
    p2_hat = p2_hat[:2].T

    residuals = np.concatenate([(p1 - p1_hat).reshape([-1]), (p2 - p2_hat).reshape([-1])])
    return residuals
    

'''
Q5.3 Extra Credit Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    r = invRodrigues(M2_init[:, :3])
    x0 = np.hstack((P_init.ravel(), r.ravel(), M2_init[:, -1]))
    
    # Nonlinear optimization for the error function above
    x, _ = leastsq(lambda x: rodriguesResidual(K1, M1, p1, K2, p2, x), x0)
    N = len(p1)
    P2 = x[:3*N].reshape(N, 3)
    M2 = np.hstack((rodrigues(x[3*N:3*N+3]), x[3*N+3:].reshape(-1,1)))

    return M2, P2