# ps2
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


# Read images
L = cv2.imread(('input\pair-L.png'), 0) * ( 1.0 / 255.0)  # grayscale
R = cv2.imread(('input\pair-R.png'), 0) * ( 1.0 / 255.0)

# Compute disparity (using method disparity_ssd defined in disparity_ssd.py)
from disparity_ncorr import disparity_ncorr
C_L = disparity_ncorr(L, R)
C_R = disparity_ncorr(R, L)



from disparity_ssd import disparity_ssd
D_L = disparity_ssd(L, R)
D_R = disparity_ssd(R, L)



def scale(image):
    return ((image - np.min(image)) * (255.0/np.max(image)) ).astype(np.uint8)

cv2.imwrite("output/ps2-1-a-1.png", scale(D_L))
cv2.imwrite("output/ps2-1-a-2.png", scale(D_R))

