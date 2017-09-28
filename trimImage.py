import cv2
import numpy as np

def main():
	imgData = cv2.imread('cooling_arm_referenced-Model.png', 0)
	print (imgData.shape)
	for i in range(imgData.shape[0]) :
		if (np.var(imgData[i]) > 0) :
			print (i)
			break
	imgData = cv2.imread('cooling_arm_referenced-Model.png')
	nimg = imgData[i - 5:]
	print (nimg.shape)
	cv2.imwrite("trimmed_image.png", nimg)


if __name__ == '__main__':
	main()