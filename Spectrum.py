import numpy as np

def getSpectrum(signal, fs = 250.0):
	fftsignal = np.fft.fft(signal)
	magnitude = np.abs(fftsignal) ** 2 / len(fftsignal)
	ndata = len(signal)
	freq = np.arange(ndata, dtype = float) / ndata * fs
	print (freq[ndata / 2])
	return {
	'x' : freq[1: ndata / 2],
	'y' : magnitude[1:ndata / 2]
	}

if __name__ == '__main__':
	pass