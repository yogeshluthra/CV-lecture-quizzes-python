import cv2
import numpy as np
from codes.helpers import *

# For Your Eyes Only
frizzy = cv2.imread('images/frizzy.png', 0)
froomer = cv2.imread('images/froomer.png', 0)
cv2.imshow('Frizzy', frizzy)
cv2.imshow('Froomer', froomer)

# Find edges in frizzy and froomer images
lTh, hTh = getOtsuThresholds(frizzy)
print lTh, hTh
frizzy_edge_Otsu = cv2.Canny(frizzy, lTh, hTh)
frizzy_edge_HardCoded = cv2.Canny(frizzy, 20, 100)
cv2.imshow('frizzy_edge_Otsu', frizzy_edge_Otsu)
cv2.imshow('frizzy_edge_HardCoded', frizzy_edge_HardCoded)

lTh, hTh = getOtsuThresholds(froomer)
print lTh, hTh
froomer_edge_Otsu = cv2.Canny(froomer, lTh, hTh)
froomer_edge_HardCoded = cv2.Canny(froomer, 20, 100)
cv2.imshow('froomer_edge_Otsu', froomer_edge_Otsu)
cv2.imshow('froomer_edge_HardCoded', froomer_edge_HardCoded)

# Display common edge pixels
common_Otsu = frizzy_edge_Otsu * froomer_edge_Otsu
cv2.imshow('Common_Otsu', common_Otsu.astype(np.float))

common_HardCoded = frizzy_edge_HardCoded * froomer_edge_HardCoded
cv2.imshow('Common_HardCoded', common_HardCoded.astype(np.float))

cv2.waitKey(0)
