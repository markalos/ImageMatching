import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('trimmed_0001.tif')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(gray,32,0.01,9, useHarrisDetector = False)
corners = np.int0(corners)

for i in corners:
	x,y = i.ravel()
	cv2.circle(img,(x,y),3,255,-1)

plt.imshow(img),plt.show()