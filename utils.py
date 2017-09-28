import math

import matplotlib.pyplot as plt
import cv2
from PIL import Image
import numpy as np

def mergeImage(backGroundFile, foreGroundFile):
	backGround = cv2.imread(backGroundFile)
	foreGround = cv2.imread(foreGroundFile)
	newImage = np.ndarray(backGround.shape)
	newImage[:, :, 0] = backGround[:, :, 0]
	newImage[:, :, 1] = foreGround[:, :, 1]
	cv2.imwrite('match.png', newImage)


def __rotate(point, rad):
	x = math.cos(rad)*point[0] - math.sin(rad)*point[1]
	point[1] = math.sin(rad)*point[0] + math.cos(rad)*point[1]
	point[0] = x
	return point

def rotatePoint(point, rad):
	return __rotate(point, rad)

def extendImage(imageFile):
	image = cv2.imread(imageFile)
	shape = image.shape
	if shape[0] < shape[1]:
		diff = shape[1] - shape[0]
		newImage = np.ndarray(shape = (shape[1], shape[1], 3))
		newImage[diff // 2 - 1] = image[0]
		for i in range(diff // 2) :
			newImage[i] = image[0]
		newImage[diff // 2 : diff // 2 + shape[0]] = image[:]
		# newImage[diff // 2 + shape[1] : ][:] = 
		for i in range(diff // 2 + shape[0], shape[1]) :
			newImage[i] = image[-1]
		cv2.imwrite('extended_' + imageFile, newImage)


def paddingToSquare(fileName, isWhite = True):
	oldImage = Image.open(fileName)
	shape = oldImage.size
	newShape = [max(shape)] * 2
	newImage = Image.new("RGB", newShape, 'white' if isWhite else 'black')
	xoffset = (newShape[0]-shape[0])//2
	yoffset = (newShape[1]-shape[1])//2
	# newImage
	newImage.paste(oldImage, (xoffset, yoffset))
	newImage.save("padded_" + fileName)

def rotatePointArray(points, rad):
	# print (points)
	points = [__rotate(p, rad) for p in points]
	return points
	# print (points)

def dotproduct(v1, v2):
	return sum((a*b) for a, b in zip(v1, v2))

def length(v):
	return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
	return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))


def opencvImageShow(image, title):
	cv2.imshow(title, image)
	if cv2.waitKey(0) & 0xff == 27:
		cv2.destroyAllWindows()

def mplImageShow(image, cmapType = 'gray'):
	cmap = plt.get_cmap(cmapType) if cmapType is not None else None
	plt.imshow(image, cmap = cmap)
	plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
	plt.show()
	

def flipImage(fileName):
	im = np.flipud(cv2.imread(fileName))
	cv2.imwrite('flipped.png', im)


def main():
	print ('utils.main')
	# extendImage('flipped.png')
	# flipImage('model.png')
	# mergeImage('scale.png', 'extended_trimmed_0001.tif')


if __name__ == '__main__':
	main()