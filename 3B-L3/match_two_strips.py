import cv2
import numpy as np
import matplotlib.pyplot as plt


# We will use the function implemented in the last quiz
# Find best match
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


def match_strips(strip_left, strip_right, b):
    # For each non-overlapping patch/block of width b in the left strip,
    # find the best matching position (along X-axis) in the right strip.
    # Return a vector of disparities (left X-position - right X-position).
    # Note: Only consider whole blocks that fit within image bounds.
    Nblocks = strip_left.shape[1]//b
    disparities=[]
    for xIndex_leftBlock in range(Nblocks):
        patch = strip_left[:, (xIndex_leftBlock)*b:(xIndex_leftBlock+1)*b]
        x_bestMatch = find_best_match(patch, strip_right)
        disparity = xIndex_leftBlock*b-x_bestMatch
        disparities.append(disparity)
    return np.asarray(disparities)[None,:]

# Test code:

# Load images
left = cv2.imread('images/flowers-left.png')
right = cv2.imread('images/flowers-right.png')
cv2.imshow('Left', left)
cv2.imshow('Right', right)

# Convert to grayscale, double, [0, 1] range for easier computation
left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY) / 255.
right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY) / 255.

# Define strip row (y) and square block size (b)
y = 94  # Adapted to mimic quiz results
b = 100

# Extract strip from left image
strip_left = left_gray[y: y + b, :]
cv2.imshow('Strip Left', strip_left)

# Extract strip from right image
strip_right = right_gray[y: y + b, :]
cv2.imshow('Strip Right', strip_right)

# Now match these two strips to compute disparity values
disparity = match_strips(strip_left, strip_right, b)
print disparity

# Finally we plot the disparity values. Note that there may be some differences
# in the results shown in the quiz because we had to adapt the index values.
plt.plot(range(disparity.shape[1]), disparity[0])
plt.show()
plt.close('all')
