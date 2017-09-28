# Harris Corner Detector in Python Thought I'd share a simple Python
# implementation of the Harris corner detector. I have seen people looking for a
# python implementation for a range of applications so I'm hoping someone finds
# this useful.

# The Harris (or Harris & Stephens) corner detection algorithm is one of the
# simplest corner indicators available. The general idea is to locate points
# where the surrounding neighborhood shows edges in more than one direction,
# these are then corners or interest points. The algorithm is explained here. In
# short, a matrix W is created from the outer product of the image gradient,
# this matrix is averaged over a region and then a corner response function is
# defined as the ratio of the determinant to the trace of W.

# Let's see what this looks like in code. First we need to be able to do
# convolutions of 2D signals. For this NumPy is not enough and we need to use
# the signal module in SciPy. Create a file filtertools.py and add the following
# functions needed to create Gaussian derivative kernels and apply them to the
# image.
from scipy import *
from scipy import signal
import cv2

import filtertools


def gauss_derivative_kernels(size, sizey=None):
	""" returns x and y derivatives of a 2D 
		gauss kernel array for convolutions """
	size = int(size)
	if not sizey:
		sizey = size
	else:
		sizey = int(sizey)
	y, x = mgrid[-size:size+1, -sizey:sizey+1]

	#x and y derivatives of a 2D gaussian with standard dev half of size
	# (ignore scale factor)
	gx = - x * exp(-(x**2/float((0.5*size)**2)+y**2/float((0.5*sizey)**2))) 
	gy = - y * exp(-(x**2/float((0.5*size)**2)+y**2/float((0.5*sizey)**2))) 

	return gx,gy

def gauss_derivatives(im, n, ny=None):
	""" returns x and y derivatives of an image using gaussian 
		derivative filters of size n. The optional argument 
		ny allows for a different size in the y direction."""

	gx,gy = cv2.getDerivKernels(n, n, 7)

	imx = signal.convolve(im,gx, mode='same')
	imy = signal.convolve(im,gy, mode='same')

	return imx,imy

def compute_harris_response(image):
	""" compute the Harris corner detector response function 
		for each pixel in the image"""

	ksize = 11

	#derivatives
	imx,imy = filtertools.gauss_derivatives(image, ksize)

	#kernel for blurring
	gauss = cv2.getGaussianKernel(ksize , 0.3*((ksize-1)*0.5 - 1) + 0.8)

	#compute components of the structure tensor
	Wxx = signal.convolve(imx*imx,gauss, mode='same')
	Wxy = signal.convolve(imx*imy,gauss, mode='same')
	Wyy = signal.convolve(imy*imy,gauss, mode='same')

	#determinant and trace
	Wdet = Wxx*Wyy - Wxy**2
	Wtr = Wxx + Wyy

	return Wdet / Wtr

# This gives an image with each pixel containing the value of the Harris
# response function. Now it is just a matter of picking out the information
# needed from this image. Picking all values above a threshold with the
# additional constraint that corners must be separated with a minimum distance
# is an approach that often gives good results. To do this, take all candidate
# pixels, sort them in descending order of corner response values and mark off
# regions too close to positions already marked as corners. Add this function to
# harris.py.

def get_harris_points(harrisim, min_distance=10, threshold=0.1):
	""" return corners from a Harris response image
		min_distance is the minimum nbr of pixels separating 
		corners and image boundary"""

	#find top corner candidates above a threshold
	corner_threshold = max(harrisim.ravel()) * threshold
	harrisim_t = (harrisim > corner_threshold) * 1

	#get coordinates of candidates
	candidates = harrisim_t.nonzero()
	coords = [ (candidates[0][c],candidates[1][c]) for c in range(len(candidates[0]))]
	#...and their values
	candidate_values = [harrisim[c[0]][c[1]] for c in coords]

	#sort candidates
	index = argsort(candidate_values)

	#store allowed point locations in array
	allowed_locations = zeros(harrisim.shape)
	allowed_locations[min_distance:-min_distance,min_distance:-min_distance] = 1

	#select the best points taking min_distance into account
	filtered_coords = []
	for i in index:
		if allowed_locations[coords[i][0]][coords[i][1]] == 1:
			filtered_coords.append(coords[i])
			allowed_locations[(coords[i][0]-min_distance):(coords[i][0]+min_distance),(coords[i][1]-min_distance):(coords[i][1]+min_distance)] = 0

	return filtered_coords

def plot_harris_points(image, filtered_coords):
	""" plots corners found in image"""
	import matplotlib.pyplot as plt

	plt.figure()
	plt.gray()
	plt.imshow(image)
	plt.plot([p[1] for p in filtered_coords],[p[0] for p in filtered_coords],'*')
	plt.axis('off')
	plt.show()

import cv2
def main():
	im = cv2.imread('trimmed_0001.tif', 0)
	harrisim = compute_harris_response(im)
	filtered_coords = get_harris_points(harrisim,6)
	plot_harris_points(im, filtered_coords)


if __name__ == '__main__':
	main()