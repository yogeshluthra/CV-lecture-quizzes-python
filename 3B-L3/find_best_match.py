import cv2
import numpy as np

def find_matched_template(patch, image, show=False):
    """returns top left of matched template w.r.t. coordinates in image"""
    assert patch.ndim == 2 and image.ndim == 2
    h, w = patch.shape
    H, W = image.shape
    image_flt32 = image.astype(np.float32)
    patch_flt32 = patch.astype(np.float32)
    normedCorrelation = cv2.matchTemplate(image_flt32, patch_flt32, method=cv2.TM_CCOEFF_NORMED)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(normedCorrelation)

    if show:
        cv2.imshow('normedCorrelation', normedCorrelation)
        matchedCenterLoc = (maxLoc[0] + w / 2, maxLoc[1] + h / 2)
        cv2.circle(image_flt32, matchedCenterLoc, 4, (255, 0, 0), -1)
        pt1 = maxLoc
        pt2 = (maxLoc[0] + w, maxLoc[1] + h)
        cv2.rectangle(image_flt32, pt1, pt2, (255, 0, 0), 2)
        cv2.imshow('matched Template', image_flt32)

    return maxLoc

# Find best match
def find_best_match(patch, strip, show=False):
    """returns only the x location where template matches strip"""
    matchedLoc = find_matched_template(patch, strip, show=False)
    return matchedLoc[0]
# Test code:

# Load images
left = cv2.imread('images/flowers-left.png')
right = cv2.imread('images/flowers-right.png')
cv2.imshow('Left', left)
cv2.imshow('Right', right)

# Convert to grayscale, double, [0, 1] range for easier computation
left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY) / 255.
right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY) / 255.

# Define image patch location (topleft [row col]) and size
patch_loc = [94, 119]  # Adapted index values to approximate the difference with the original images shapes
patch_size = [100, 100]

# Extract patch (from left image)
patch_left = left_gray[patch_loc[0]:patch_loc[0] + patch_size[0],
                       patch_loc[1]:patch_loc[1] + patch_size[1]]
cv2.imshow('Patch', patch_left)

# Extract strip (from right image)
strip_right = right_gray[patch_loc[0]: patch_loc[0] + patch_size[0], :]
cv2.imshow('Strip', strip_right)

# Now look for the patch in the strip and report the best position (column index of topleft corner)
best_x = find_best_match(patch_left, strip_right)
print best_x

patch_right = right_gray[patch_loc[0]: patch_loc[0] + patch_size[0],
                         best_x: best_x + patch_size[1]]

cv2.waitKey(0)
cv2.destroyAllWindows()
