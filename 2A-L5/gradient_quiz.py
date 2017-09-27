import cv2
import numpy as np
import scipy.signal as sp
import time


# Helper functions
def normalize(img_in):
    img_out = np.zeros(img_in.shape)
    cv2.normalize(img_in, img_out, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return img_out

def normalize_image(image_in):
    """normalize the image between 0-255 and setting data type=uint8"""
    minImg, maxImg = np.min(image_in) * 1., np.max(image_in) * 1.
    return (((image_in - minImg) / (maxImg - minImg)) * 255).astype(np.uint8)

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

# Gradient Direction
def select_gdir(gmag, gdir, mag_min, angle_low, angle_high):
    # TODO: Find and return pixels that fall within the desired mag, angle range
    pix = np.zeros(gmag.shape, dtype=np.uint8)
    pix[np.bitwise_and(gdir >= mag_min,
                        np.bitwise_and(gdir >= angle_low, gdir < angle_high))] \
        = 255
    return pix

# Load and convert image to double type, range [0, 1] for convenience
img = cv2.imread('images/octagon.png', 0) / 255.
cv2.imshow('Image', img)  # assumes [0, 1] range for double images

# Compute x, y gradients
# TODO: Check out http://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=sobel#cv2.Sobel
# gx = cv2.Sobel(img, -1, dx=1, dy=0)
# gy = cv2.Sobel(img, -1, dx=0, dy=1)
gx,gy = imgradientxy(img)
cv2.imshow('Gx', gx)
cv2.imshow('Gy', gy)

gmag, gdir = imgradient(gx,gy)

cv2.imshow('Gmag1', normalize(gmag).astype(np.uint8))
cv2.imshow('Gdir1', normalize(gdir).astype(np.uint8))

cv2.imshow('Gmag2', normalize_image(gmag).astype(np.uint8))
cv2.imshow('Gdir2', normalize_image(gdir).astype(np.uint8))

# Find pixels with desired gradient direction
my_grad = select_gdir(gmag, gdir, 1, 30, 60)
cv2.imshow('My Grad', my_grad)
cv2.waitKey(0)