import cv2

# Explore edge options

# Load an image
img = cv2.imread('images/fall-leaves.png')
cv2.imshow('Image', img)

# - set filter specifics
filter_size=11
filter_spatial_sigma=2

# TODO: Create a Gaussian filter. Use cv2.getGaussianKernel.
gauss_filter=cv2.getGaussianKernel(filter_size, filter_spatial_sigma)

# TODO: Apply it, specifying an edge parameter (try different parameters). Use cv2.filter2D.
# - just simple pladding
img_smoothed_0padded = cv2.sepFilter2D(img, -1, gauss_filter, gauss_filter, borderType=cv2.BORDER_CONSTANT)
cv2.imshow('img_smoothed_0padded', img_smoothed_0padded)

# - reflected image at border
img_smoothed_reflected = cv2.sepFilter2D(img, -1, gauss_filter, gauss_filter, borderType=cv2.BORDER_REFLECT)
cv2.imshow('img_smoothed_reflected', img_smoothed_reflected)

# - wrapped image at border. Some issues with cv2.BORDER_WRAP. Look at Pedro's answers for details on issue.
img_wrapped = cv2.copyMakeBorder(img, filter_size, filter_size, filter_size, filter_size, cv2.BORDER_WRAP)
cv2.imshow('img_wrapped', img_wrapped)
img_smoothed_wrapped = cv2.sepFilter2D(img_wrapped, -1, gauss_filter, gauss_filter)
img_smoothed_wrapped = img_smoothed_wrapped[filter_size:-filter_size, filter_size:-filter_size]
cv2.imshow('img_smoothed_wrapped', img_smoothed_wrapped)

# - replicate (simply extend the edges)
img_smoothed_replicated = cv2.sepFilter2D(img, -1, gauss_filter, gauss_filter, borderType=cv2.BORDER_REPLICATE)
cv2.imshow('img_smoothed_replicated', img_smoothed_replicated)

cv2.waitKey(0)
cv2.destroyAllWindows()