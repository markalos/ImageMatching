import numpy as np
import matplotlib.pyplot as plt
import cv2

import utils


def distanceEuclidean(a, b):
	return np.linalg.norm(a-b)

def plotWithoutShow(points, color = "red"):
	def getX(points):
		return [p[0] for p in points]
	def getY(points):
		return [p[1] for p in points]
	
	plt.plot(getX(points), getY(points), marker='o', markersize=3, color=color)
	
	

def sortPointOnCircle(points):
	if len(points) < 4:
		return points
	points.sort(key = lambda p : p[0])
	offset = points[0]
	npoints = points[1:]
	def cosKey(point):
		xdiff = point[0] - offset[0]
		ydiff = point[1] - offset[1]
		return (- ydiff) / np.sqrt(xdiff * xdiff + ydiff * ydiff)
	npoints.sort(key = cosKey)
	return ( [offset] + npoints)

def getSortedPoints(fname) :
	points = []
	with open(fname) as src :
		for line in src :
			# line.
			points.append([float(d) for d in line.rstrip().split(',')])
	points = sortPointOnCircle(points)
	return points
	

def getOffset(fname):
	points = getSortedPoints(fname)
	points = np.array(points)
	# print (points)
	center = (np.mean(points, axis=0))
	points = points - center
	return (points, center)
	# print (center)
	# distances = [distanceEuclidean(p, center) for p in points]
	# print (fname,np.mean(distances))

def getR(points) :
	return np.mean([distanceEuclidean([0, 0], p) for p in points])


def translate(imgFile, fromPoint, toPoint, targetSize):
	img = cv2.imread(imgFile)
	pts1 = np.float32(fromPoint)
	pts2 = np.float32(toPoint)
	M = cv2.getPerspectiveTransform(pts1,pts2)

	print (targetSize)
	dst = cv2.warpPerspective(img,M, dsize = targetSize)
	plt.subplot(121),plt.imshow(img),plt.title('Input')
	plt.subplot(122),plt.imshow(dst),plt.title('Output')
	plt.show()

def scaleImage(rad, scale, bpcenter, semcenter, image, newShape):
	newImage = np.ndarray(newShape, dtype = int)
	# newCenter = center + (newShape[0] - image.shape[0]) // 2
	lim = image.shape[0] - 1
	def getCorrespondPoint(p) :
		mapto = (np.array([p[0] - semcenter[0], p[1] - semcenter[1]]) / scale)
		mapto = utils.rotatePoint(mapto, rad) + bpcenter
		# print (mapto)
		return np.clip(mapto, 0, lim)
	for i in range(newShape[0]) :
		for j in range(newShape[1]) :
			npoint = getCorrespondPoint([i,j]).astype(int)
			# print (npoint)
			newImage[i][j] = image[npoint[0]][npoint[1]]
	cv2.imwrite("scale.png", newImage)

def main():
	files = ["Locations.txt",
	"BluePrintLocations.txt"
	]
	sempoints, semcenter = getOffset(files[0])
	bppoints, bpcenter = getOffset(files[1])
	# targetSize = cv2.imread('extended_trimmed_0001.tif', 0).shape
	# fromPoint = (sempoints[:4])
	# toPoint = (bppoints[:4])
	# translate('extended_model.png', fromPoint, toPoint, targetSize)
	semr = getR(sempoints)
	bpr = getR(bppoints)
	scale = semr / bpr
	bppoints = bppoints * scale
	rad = utils.angle(sempoints[0], bppoints[0])
	bppoints = utils.rotatePointArray(bppoints, -rad)
	semImage = cv2.imread('extended_trimmed_0001.tif')
	bpImage = cv2.imread('extended_flipped.png')
	scaleImage(rad, scale, bpcenter, semcenter, bpImage, semImage.shape)
	utils.mergeImage('scale.png', 'extended_trimmed_0001.tif')
	# plotWithoutShow(sempoints + semcenter)
	# plotWithoutShow(bppoints + semcenter, "blue")
	# plt.imshow(semImage)
	# plt.savefig('attempt.png', transparent=True, bbox_inches='tight', pad_inches=0)
	# plt.show()


def objectGetq():
	return {
	'x' :0,
	'y' : 1
	}


if __name__ == '__main__':
	main()
