from submission import *
from helper import *
from os.path import exists
import numpy as np
import cv2

'''
Q2 - Q4 Testing
'''
# data = np.load('../data/some_corresp.npz')
# pts1 = data['pts1']
# pts2 = data['pts2']
# I1 = cv2.imread("../data/im1.png")
# I2 = cv2.imread("../data/im2.png")
# M = max(I1.shape[0], I1.shape[1])

# F = eightpoint(pts1, pts2, M)
# np.savez('q2_1.npz', F=F, M=M)
# displayEpipolarF(I1[:, :, ::-1], I2[:, :, ::-1], F)

# intrinsics = np.load('../data/intrinsics.npz')
# K1, K2 = intrinsics['K1'], intrinsics['K2']
# E = essentialMatrix(F, K1, K2)
# np.savez('q3_1.npz', E=E)

# epipolarMatchGUI(I1[:, :, ::-1], I2[:, :, ::-1], F)

'''
Ransac F
'''
# data = np.load('../data/some_corresp_noisy.npz')
# pts1, pts2 = data['pts1'], data['pts2']
# I1 = cv2.imread("../data/im1.png")
# I2 = cv2.imread("../data/im2.png")
# M = max(I1.shape[0], I1.shape[1])

# F, inliers = ransacF(pts1, pts2, M, 200, tol=4.5)
# print("Number of Inliers: %s (Reference: 106)" % np.sum(inliers))
# # np.savez('q5_1.npz', F=F, M=M, inliers=inliers)
# displayEpipolarF(I1[:, :, ::-1], I2[:, :, ::-1], F)

'''
Bundle Adjustment
'''
data = np.load('../data/some_corresp_noisy.npz')
pts1, pts2 = data['pts1'], data['pts2']
I1 = cv2.imread("../data/im1.png")
I2 = cv2.imread("../data/im2.png")
M = max(I1.shape[0], I1.shape[1])

if exists('q5_1.npz'):
    ransac = np.load('q5_1.npz')
    F, inliers = ransac['F'], ransac['inliers']
else: 
    F, inliers = ransacF(pts1, pts2, M, 500, tol=1.3)
    np.savez('q5_1.npz', F=F, M=M, inliers=inliers)
print("Number of Inliers: %s (Reference: 106)" % np.sum(inliers))
# displayEpipolarF(I1[:, :, ::-1], I2[:, :, ::-1], F)
p1, p2 = pts1[inliers], pts2[inliers]

intrinsics = np.load('../data/intrinsics.npz')
K1, K2 = intrinsics['K1'], intrinsics['K2']
E = essentialMatrix(F, K1, K2)

# Building initialization for optimization
M1 = np.zeros((3,4))
M1[:3,:3] = np.eye(3)
C1 = K1 @ M1

M2s = camera2(E)
minErr = float('inf')
for i in range(M2s.shape[2]):
    C2 = K2 @ M2s[:,:,i]
    P, err = triangulate(C1, p1, C2, p2)

    if np.all(P[:, -1] > 0):
        if err < minErr:
            bestP = P
            minErr = err
            print('Reprojection Error before Optimization: ', err)
            M2 = M2s[:,:,i]

M2, P2 = bundleAdjustment(K1, M1, p1, K2, M2, p2, bestP)
C2 = K2 @ M2  # Update C2 with optimized M2 
_, err_star = triangulate(C1, p1, C2, p2)
print('Reprojection Error after Optimization: ', err_star)

fig = plt.figure()
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.scatter(bestP[:, 0], bestP[:, 1], bestP[:, 2], s=3, c='r')
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
ax.set_title('Original 3D Points')
ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.scatter(P2[:, 0], P2[:, 1], P2[:, 2], s=3, c='g')
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
ax.set_title('Optimized 3D Points')
plt.show()