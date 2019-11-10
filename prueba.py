import scipy.io as scp
import numpy as np
import matplotlib.pyplot as plt

CANTIDAD = 200

datamat = scp.loadmat('Datos/103m.mat')

datamat_y = datamat['y']
y = np.array(datamat_y)[0]

marcas = np.loadtxt('Datos/marcas.txt')

print(len(marcas), marcas)

for m in marcas:
	plt.axvline(x=m, color='red')



print(y)

plt.plot(y)
plt.show()