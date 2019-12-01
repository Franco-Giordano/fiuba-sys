from ecg import *

import math

derivada = np.loadtxt("ecg_filtrado_hl_hh_hd.out")


al_cuadrado = np.power(derivada, 2)

plotear(al_cuadrado[100:800])

fft = fft(al_cuadrado,2048)  # fft de 2048 puntos
freq = np.linspace(-math.pi, math.pi, len(fft))
response = 20 * np.log10(np.abs(fftshift(fft/abs(fft).max())))

sinDB = abs(fftshift(fft))
plt.grid()
plt.plot(freq,sinDB)

plt.show()

np.savetxt("ecg_filtrado_hl_hh_hd_cuadrado.out", al_cuadrado)