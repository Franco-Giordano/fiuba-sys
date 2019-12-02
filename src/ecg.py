import scipy.io as scp
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fftpack import fft
from scipy.fftpack import fftshift

__datamat = scp.loadmat('../data/103m.mat')
__datamat_y = __datamat['y']
__z = np.array(__datamat_y)[0]

ecg_original = __z - __z.mean()

ecg = ecg_original/200

def plotear(xs, color=None, show=True, label=''):
	#plt.figure()
	if label:
		if color:
			plt.plot(xs,color=color, label=label)
		else:
			plt.plot(xs,label=label)
	else:
		if color:
			plt.plot(xs,color=color)
		else:
			plt.plot(xs)
		
	if show:
		plt.legend()
		plt.show()
	return

if __name__ == '__main__':
	plt.axvspan(215000, 331000, color='r', alpha=0.5, lw=0)
	plt.xlabel("Numero de muestra")
	plt.ylabel("Amplitud [mV]")
	plt.title("Derivacion completa, destacando en rojo la seccion de un ruido no estacionario")
	plotear(ecg)