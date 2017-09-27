import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# - HELPERs
def normalize_image(image):
    """normalize the image between 0-255 and setting data type=uint8"""
    minImg, maxImg = np.min(image)*1., np.max(image)*1.
    return (((image - minImg)/(maxImg-minImg))*255).astype(np.uint8)

def imshow(title, image):
    """normalize the image between 0-255 and then show"""
    print 'To show, first normalizing the image between 0-255 and then setting data type=uint8'
    image = normalize_image(image)
    cv2.imshow(title, image)

def imgradientxy(img):
    """find gradients in x and y directions"""
    gx = cv2.Sobel(img, -1, dx=1, dy=0)
    gy = cv2.Sobel(img, -1, dx=0, dy=1)
    return gx, gy

def imgradient(gx,gy):
    """
    gmag = (gx^2 + gy^2)/(4*sqrt(2)) \n
    gdir = atan(-gy/gx)/pi * 180
    """
    # The minus sign here is used based on how imgradient is implemented in octave
    # See https://sourceforge.net/p/octave/image/ci/default/tree/inst/imgradient.m#l61
    # - reason answered here https://stackoverflow.com/questions/18549015/why-imgradient-invert-vertical-when-computing-the-angle
    #   - A typical coordinate system has its origin in the bottom left. A Matlab matrix has the origin in the top left. Inverting the y-coordinate may compensate for that.
    gmag = np.sqrt(gx ** 2 + gy ** 2) / (4 * np.sqrt(2))
    gdir = np.arctan2(-gy, gx) * 180 / np.pi
    return gmag, gdir

def plot_stat_3D(img, ax=None, color='r'):
    Y, X = map(np.arange, img.shape)
    X, Y = np.meshgrid(X, Y)
    if ax is None: ax=Axes3D(plt.figure())
    ax.plot_wireframe(X, Y, img, alpha=0.5, color=color)
    # plt.show()
    return ax

def plot_stat_countourf(img):
    Y, X = map(np.arange, img.shape)
    X, Y = np.meshgrid(X, Y)
    levels=np.linspace(np.min(img), np.max(img), 10)
    plt.contourf(X, Y, img, levels, cmap=plt.cm.jet)
    plt.colorbar()
    plt.show()

def plot_stat_3D_rotate(img):
    Y, X = map(np.arange, img.shape)
    X, Y = np.meshgrid(X, Y)
    ax=Axes3D(plt.figure())
    ax.plot_wireframe(X, Y, img)
    # rotate the axes and update
    for angle in range(0, 360):
        ax.view_init(30, angle)
        plt.draw()
        plt.pause(.1)

def getOtsuThresholds(img):
    assert img.dtype==np.uint8
    high_thresh, thresh_im = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_thresh = 0.5 * high_thresh
    return low_thresh, high_thresh