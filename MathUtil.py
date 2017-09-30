import math
import numpy as np

def dotproduct(v1, v2):
	return sum((a*b) for a, b in zip(v1, v2))

def rotateArray(arr, n):
	n %= len(arr)
	if 0 == n :
		return arr[:]
	return arr[n :] + arr[ : n]

def length(v):
	return math.sqrt(dotproduct(v, v))

def getRotateAngle(v1, v2):
	return math.acos(np.clip(dotproduct(v1, v2) / (length(v1) * length(v2)), -1, 1))

def distanceEuclidean(a, b):
	return np.linalg.norm(sum(x1 - x2 for x1, x2 in zip(a, b)))

def mean(numbers):
	return float(sum(numbers)) / max(len(numbers), 1)

def RMSE(v1, v2):
	return mean([distanceEuclidean(a, b) for a, b in zip(v1, v2)])

def rotateCore(point, rad):
	x = math.cos(rad)*point[0] - math.sin(rad)*point[1]
	y = math.sin(rad)*point[0] + math.cos(rad)*point[1]
	return [x, y]

def rotatePoint(point, rad):
	return rotateCore(point, rad)

def rotatePointArray(points, rad):
	# print (points)
	points = [rotateCore(p, rad) for p in points]
	return points

def minIndex(data):
	pairs = [(i, v) for i, v in enumerate(data)]
	return min(pairs, key = lambda p : p[1])[0]

def getBestMatchAngle(ref, points):
	rads = []
	errors = []
	for i in range(len(points)) :
		npoints = rotateArray(points, i)
		rad = getRotateAngle(ref[0], npoints[0])
		rads.append(rad)
		errors.append(RMSE(ref, rotatePointArray(npoints, rad)))
		rads.append(-rad)
		errors.append(RMSE(ref, rotatePointArray(npoints, -rad)))

	# print (errors)
	# print (rads)
	return rads[minIndex(errors)]

def testDriver():
	
	angles = [math.pi * (0.2 + i / 12.0)  for i in range(12)]
	points = [[math.cos(angle), math.sin(angle)] for angle in angles]
	ref = rotatePointArray(points, math.pi * 0.35)
	print (math.pi * 0.35)
	rad = getBestMatchAngle(rotateArray(ref, 6), points)
	print (rad)

if __name__ == '__main__':
	testDriver()
