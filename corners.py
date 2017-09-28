import cv2
import numpy as np
import os

import utils


def main():
	filename = 'trimmed_0001.tif'
	dsfName = os.path.splitext(filename)[0] + '_harriCorners.png'
	print (dsfName)
	img = cv2.imread(filename)
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	gray = np.float32(gray)
	dst = cv2.cornerHarris(gray, blockSize = 2, ksize = 9, k = 0.01)
	#result is dilated for marking the corners, not important
	dst = cv2.dilate(dst,None)

	# Threshold for an optimal value, it may vary depending on the image.
	img[dst>0.1*dst.max()]=[0,0,255]

	cv2.imwrite(dsfName, img)
	utils.opencvImageShow(img, 'corners')

if __name__ == '__main__':
	main()