'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''

import numpy as np
import cv2
from submission import triangulate, eightpoint, essentialMatrix
from helper import camera2

data = np.load('../data/some_corresp.npz')
pts1 = data['pts1']
pts2 = data['pts2']
I1 = cv2.imread("../data/im1.png")
I2 = cv2.imread("../data/im2.png")
M = max(I1.shape[0], I1.shape[1])
intrinsics = np.load('../data/intrinsics.npz')
K1, K2 = intrinsics['K1'], intrinsics['K2']

F = eightpoint(pts1, pts2, M)
# print(F)
E = essentialMatrix(F, K1, K2)
# print(E)

M1 = np.zeros((3,4))
M1[:3,:3] = np.eye(3)
C1 = K1 @ M1

M2s = camera2(E)
minErr = float('inf')
for i in range(M2s.shape[2]):
    C2 = K2 @ M2s[:,:,i]
    P, err = triangulate(C1, pts1, C2, pts2)

    if np.all(P[:, -1] > 0):
        if err < minErr:
            bestP = P
            minErr = err
            print('Reprojection Error: ', err)
            np.savez('q3_3.npz', M2=M2s[:,:,i], C2=C2, P=P)