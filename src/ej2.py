from ecg import *

import math

#transformada de fourier, modulo, de un qrs

unQRS = ecg[312:329]

fft = fft(unQRS,2048)  # fft de 2048 puntos
freq = np.linspace(-100, 100, len(fft))
response = 20 * np.log10(np.abs(fftshift(fft/abs(fft).max())))

sinDB = abs(fftshift(fft))

plt.title("Respuesta en frecuencia de un complejo QRS")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Amplitud")
plt.plot(freq,sinDB)

plt.show()


#un qrs en tiempo

plt.figure()
plt.title("Complejo QRS")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [mV]")


tiempo = np.linspace(0, len(unQRS)/200, len(unQRS))
plt.plot(tiempo,unQRS)

plt.show()