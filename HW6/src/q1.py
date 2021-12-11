import numpy as np
from matplotlib import pyplot as plt
from skimage.io import imread
from skimage.color import rgb2xyz
from utils import integrateFrankot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

def renderNDotLSphere(center, rad, light, pxSize, res):

    """
    Question 1 (b)

    Render a sphere with a given center and radius. The camera is 
    orthographic and looks towards the sphere in the negative z
    direction. The camera's sensor axes are centerd on and aligned
    with the x- and y-axes.

    Parameters
    ----------
    center : numpy.ndarray
        The center of the sphere in an array of size (3,)

    rad : float
        The radius of the sphere

    light : numpy.ndarray
        The direction of incoming light

    pxSize : float
        Pixel size

    res : numpy.ndarray
        The resolution of the camera frame

    Returns
    -------
    image : numpy.ndarray
        The rendered image of the sphere
    """
    if center[0] or center[1] or center[2]:
        raise ValueError("Warning: Sphere not at the center!")
    
    image = np.zeros((res[::-1]))  # resolution is given as width x height
    col, row = np.meshgrid(np.arange(res[0]), np.arange(res[1]))
    imgW, imgH = res * pxSize
    X = col * pxSize - imgW / 2
    Y = row * pxSize - imgH / 2
    sphereInd = X**2 + Y**2 <= rad**2
    X = X[sphereInd]
    Y = Y[sphereInd]
    Z = np.sqrt(rad**2 - X**2 - Y**2)
    
    # Surface normal of a sphere is just (x,y,z) / rad
    # n dot l
    I = (X * light[0] + Y * light[1] + Z * light[2]) / rad
    I = np.clip(I, 0, None)
    image[sphereInd] = I

    return image


def loadData(path = "../data/"):

    """
    Question 1 (c)

    Load data from the path given. The images are stored as input_n.tif
    for n = {1...7}. The source lighting directions are stored in
    sources.mat.

    Paramters
    ---------
    path: str
        Path of the data directory

    Returns
    -------
    I : numpy.ndarray
        The 7 x P matrix of vectorized images

    L : numpy.ndarray
        The 3 x 7 matrix of lighting directions

    s: tuple
        Image shape

    """

    I = []
    img = None
    for i in range(1, 8):
        img = imread(path + 'input_%s.tif' % i).astype(np.uint16)
        img = rgb2xyz(img)[:,:,1]  # extract the luminance channel
        I.append(img.ravel())
    L = np.load(path + 'sources.npy').T
    s = img.shape[:2]

    return np.array(I), L, s


def estimatePseudonormalsCalibrated(I, L):

    """
    Question 1 (e)

    In calibrated photometric stereo, estimate pseudonormals from the
    light direction and image matrices

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P array of vectorized images

    L : numpy.ndarray
        The 3 x 7 array of lighting directions

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals
    """

    B = np.linalg.lstsq(L.T, I, rcond=None)[0]
    return B


def estimateAlbedosNormals(B):

    '''
    Question 1 (e)

    From the estimated pseudonormals, estimate the albedos and normals

    Parameters
    ----------
    B : numpy.ndarray
        The 3 x P matrix of estimated pseudonormals

    Returns
    -------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals
    '''

    albedos = np.linalg.norm(B, axis=0)
    normals = B / albedos
    return albedos, normals


def displayAlbedosNormals(albedos, normals, s):

    """
    Question 1 (f)

    From the estimated pseudonormals, display the albedo and normal maps

    Please make sure to use the `coolwarm` colormap for the albedo image
    and the `rainbow` colormap for the normals.

    Parameters
    ----------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    -------
    albedoIm : numpy.ndarray
        Albedo image of shape s

    normalIm : numpy.ndarray
        Normals reshaped as an s x 3 image

    """

    albedoIm = albedos.reshape(s)
    normalIm = normals.T.reshape(s[0], s[1], 3)
    normalized_normalIm = normalIm - np.min(normalIm)
    normalized_normalIm /= np.max(normalized_normalIm)
    
    plt.figure()
    plt.imshow(albedoIm, cmap='gray')
    plt.show()
    plt.close()
    
    plt.figure()
    plt.imshow(normalized_normalIm, cmap='rainbow')
    plt.show()

    return albedoIm, normalIm


def estimateShape(normals, s):

    """
    Question 1 (i)

    Integrate the estimated normals to get an estimate of the depth map
    of the surface.

    Parameters
    ----------
    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    ----------
    surface: numpy.ndarray
        The image, of size s, of estimated depths at each point

    """
    fx = (- normals[0] / normals[2]).reshape(s)
    fy = (- normals[1] / normals[2]).reshape(s)
    surface = integrateFrankot(fx, fy)
    return surface


def plotSurface(surface):

    """
    Question 1 (i) 

    Plot the depth map as a surface

    Parameters
    ----------
    surface : numpy.ndarray
        The depth map to be plotted

    Returns
    -------
        None

    """
    X, Y = np.meshgrid(np.arange(surface.shape[1]), np.arange(surface.shape[0]))
    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_surface(X, Y, surface, cmap='coolwarm', linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)  # add a colorbar
    # ax.view_init(elev=-129, azim=-118)
    plt.show()

    # Generate GIF for Demo
    # def rotate(angle):
    #     ax.view_init(elev=-140, azim=angle)
    # rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0,362,2),interval=40)
    # rot_animation.save('../results/demo.gif', dpi=200, writer='imagemagick')


if __name__ == '__main__':

    # Q1b
    light = np.array([[1,1,1], [1,-1,1], [-1,-1,1]]) / np.sqrt(3)
    center = np.array([0,0,0])
    rad = 0.75
    pxSize = 7e-4  # in cm
    res = np.array([3840, 2160])
    for i in range(len(light)):
        image = renderNDotLSphere(center, rad, light[i], pxSize, res)
        plt.figure()
        plt.imshow(image, cmap='gray')
        plt.title('Rendering with %s' % light[i])
        plt.show()

    # Q1d
    I, L, s = loadData()
    _, S, _ = np.linalg.svd(I, full_matrices=False)
    print(S)

    # Q1f
    B = estimatePseudonormalsCalibrated(I, L)
    albedos, normals = estimateAlbedosNormals(B)
    albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)

    # Q1h
    surface = estimateShape(normals, s)
    plotSurface(surface)