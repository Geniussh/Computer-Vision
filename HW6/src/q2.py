# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# April 20, 2020
# ##################################################################### #

import numpy as np
from q1 import loadData, estimateAlbedosNormals, displayAlbedosNormals
from q1 import estimateShape, plotSurface 
from utils import enforceIntegrability
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def estimatePseudonormalsUncalibrated(I):

    """
    Question 2 (b)

    Estimate pseudonormals without the help of light source directions. 

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P matrix of loaded images

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals

    """
    u, s, vh = np.linalg.svd(I, full_matrices=False)
    s = np.diag(np.sqrt(s[:3]))
    L = (u[:, :3] @ s).T
    B = s @ vh[:3]

    # Normalize L so that it's consistent with the ground truth
    L /= np.linalg.norm(L, axis=0)

    return B, L


if __name__ == "__main__":
    # Q2b
    I, L0, s = loadData()
    print("Ground Truth Lighting Directions:")
    print(L0)
    B, L = estimatePseudonormalsUncalibrated(I)
    print("Estimated Lighting Directions:")
    print(L)
    albedos, normals = estimateAlbedosNormals(B)
    albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)

    # Q2d
    surface = estimateShape(normals, s)
    plotSurface(surface)

    # Q2e
    Bt = enforceIntegrability(B, s)
    _, normals = estimateAlbedosNormals(Bt)
    surface = estimateShape(normals, s)
    plotSurface(surface)

    # Q2f: bas-relief
    rs = np.array([[-1, 0, 1], [1, 0, 1],
                  [0, -1, 1], [0, 1, 1],
                  [0, 0, 0.1], [0, 0, 2]])
    for r in rs:
        G = np.eye(3)
        G[2] = r
        Bamb = np.linalg.inv(G).T @ Bt
        _, normals = estimateAlbedosNormals(Bamb)
        surface = estimateShape(normals, s)
        
        X, Y = np.meshgrid(np.arange(surface.shape[1]), np.arange(surface.shape[0]))
        fig = plt.figure()
        ax = Axes3D(fig)
        surf = ax.plot_surface(X, Y, surface, cmap='coolwarm', linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)  # add a colorbar
        ax.view_init(elev=-129, azim=-118)
        fig.suptitle(r'$\mu$={}, $\nu$={}, $\lambda$={}'.format(*r))
        plt.show()