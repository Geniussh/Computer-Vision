'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''

from submission import *
from helper import *
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import animation

data = np.load('../data/some_corresp.npz')  # use this to calcualte F and E
intrinsics = np.load('../data/intrinsics.npz')
K1, K2 = intrinsics['K1'], intrinsics['K2']
pts1 = data['pts1']
pts2 = data['pts2']
I1 = cv2.imread("../data/im1.png")
I2 = cv2.imread("../data/im2.png")
M = max(I1.shape[0], I1.shape[1])

F = eightpoint(pts1, pts2, M)
E = essentialMatrix(F, K1, K2)

# Load the actual correspondences for point cloud
coords = np.load('../data/templeCoords.npz')
x1s, y1s = coords['x1'], coords['y1']
x1s, y1s = x1s.ravel(), y1s.ravel()
x2s, y2s = [], []
for x1, y1 in zip(x1s, y1s):
    x2, y2 = epipolarCorrespondence(I1, I2, F, x1, y1)
    x2s.append(x2)
    y2s.append(y2)
x2s = np.array(x2s)
y2s = np.array(y2s)

pts1 = np.zeros((len(x1s), 2))
pts2 = np.zeros((len(x2s), 2))
pts1[:, 0], pts1[:, 1] = x1s, y1s
pts2[:, 0], pts2[:, 1] = x2s, y2s

M1 = np.zeros((3,4))
M1[:3,:3] = np.eye(3)
C1 = K1 @ M1

M2s = camera2(E)
minErr = float('inf')
bestC2 = None
for i in range(M2s.shape[2]):
    C2 = K2 @ M2s[:,:,i]
    P, err = triangulate(C1, pts1, C2, pts2)

    if np.all(P[:, -1] > 0):
        if err < minErr:
            bestP = P
            minErr = err
            print('Reprojection Error: ', err)
            M2 = M2s[:,:,i]
            bestC2 = C2

C2 = bestC2
np.savez('q4_2.npz', F=F, M1=M1, C1=C1, M2=M2, C2=C2)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.scatter(bestP[:, 0], bestP[:, 1], bestP[:, 2], s=3)

# Generate GIF for Demo
# def rotate1(angle):
#     ax.view_init(elev=angle, azim=180)
# rot_animation = animation.FuncAnimation(fig, rotate1, frames=np.arange(-180,182,2),interval=40)
# rot_animation.save('../result/rotation1.gif', dpi=200, writer='imagemagick')
# def rotate2(angle):
#     ax.view_init(elev=-85, azim=angle)
# rot_animation = animation.FuncAnimation(fig, rotate2, frames=np.arange(0,362,2),interval=40)
# rot_animation.save('../result/rotation2.gif', dpi=200, writer='imagemagick')

plt.show()