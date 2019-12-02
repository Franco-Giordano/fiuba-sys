from ecg import *

import math

derivada = np.loadtxt("ecg_filtrado_hl_hh_hd.out")


al_cuadrado = np.power(derivada, 2)

tiempo = np.linspace(0, len(al_cuadrado)/200, len(al_cuadrado))
plt.plot(tiempo[100:800],al_cuadrado[100:800])
plt.show()

fft = fft(al_cuadrado,2048)  # fft de 2048 puntos
freq = np.linspace(-100, 100, len(fft))
response = 20 * np.log10(np.abs(fftshift(fft/abs(fft).max())))

sinDB = abs(fftshift(fft))
plt.grid()
plt.plot(freq,sinDB)

plt.show()

np.savetxt("ecg_filtrado_hl_hh_hd_cuadrado.out", al_cuadrado)