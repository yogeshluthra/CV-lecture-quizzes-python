import cv2
import numpy as np


# Helper function
def imnoise(img_in, method, dens):

    if method == 'salt & pepper':
        img_out = np.copy(img_in)
        r, c = img_in.shape
        x = np.random.rand(r, c)
        black_mask = x < dens/2.
        img_out[black_mask] = 0
        white_mask = 1.-x < dens/2.
        img_out[white_mask] = 255

        return img_out

    else:
        print "Method {} not yet implemented.".format(method)
        exit()

# Apply a median filter

# Read an image
img = cv2.imread('images/moon.png', 0)
cv2.imshow('Image', img)

# TODO: Add salt & pepper noise
img_noisy=imnoise(img, 'salt & pepper', 0.02)
cv2.imshow('img_noisy', img_noisy)

# TODO: Apply a median filter. Use cv2.medianBlur
img_smoothed = cv2.medianBlur(img_noisy, 3)
cv2.imshow('img_smoothed',img_smoothed)

cv2.waitKey(0)
cv2.destroyAllWindows()
