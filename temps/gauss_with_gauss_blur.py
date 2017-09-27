
import cv2
import numpy as np
from codes.helpers import *

ax=Axes3D(plt.figure())

g1=cv2.getGaussianKernel(31, 5)
g1=g1.dot(g1.T)
print np.sum(g1)
plot_stat_3D(g1, ax=ax, color='r')

g2=cv2.getGaussianKernel(11,2)
g2=g2.dot(g2.T)
print np.sum(g2)

g3=cv2.filter2D(g1, -1, g2, borderType=cv2.BORDER_WRAP)
print np.sum(g3)
plot_stat_3D(g3, ax=ax, color='b')

plt.show()
