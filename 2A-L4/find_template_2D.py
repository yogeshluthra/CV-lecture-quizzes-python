import cv2
import numpy as np
import scipy as sp

"""CHECKOUT: http://docs.opencv.org/2.4/doc/tutorials/imgproc/histograms/template_matching/template_matching.html"""

# Find template 2D
def find_template_2D(template, img):
    # TODO: Find template in img and return [y x] location. Make sure this location is the top-left corner of the match.
    result=cv2.matchTemplate(img, template, cv2.TM_CCORR_NORMED) # normed correlation
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result) # returns location in (width, height)=(x, y) format
    return maxLoc

tablet = cv2.imread('images/tablet.png', 0)
cv2.imshow('Tablet', tablet)

glyph = tablet[74:165, 149:184]
cv2.imshow('Glyph', glyph)

# tablet_2 = 1. * tablet - np.mean(tablet)
# glyph_2 = 1. * glyph - np.mean(glyph)
# y, x = find_template_2D(glyph_2, tablet_2)

x, y = find_template_2D(glyph, tablet) # gets result in (width, height) = (x, y) format
print "Y: {}, X: {}".format(y, x)

# The code below is not part of the quiz but helps to verify the results are
# what we are looking for.
tablet = cv2.cvtColor(tablet, code=cv2.COLOR_GRAY2BGR)
cv2.rectangle(tablet, (x, y), (x + glyph.shape[1], y + glyph.shape[0]),
              (0, 0, 255), thickness=2)
cv2.imshow('Rectangle', tablet)
cv2.waitKey(0)
