from ecg import *


cuatro_latidos = ecg[100:800]

#plotear(cuatro_latidos)

fs = 200

window = signal.windows.tukey(55) # ventana de Tukey de 256 muestras
plt.figure(0)
#plotear(window)
f, t, Sxx = signal.spectrogram(cuatro_latidos, fs, window)
plt.pcolormesh(t,f,Sxx)
plt.title("Espectrograma de 4 latidos, con ventana de 55 muestras")
plt.ylabel('Frecuencia [Hz]')
plt.xlabel('Tiempo [sec]')
plt.show()


plt.figure(1)
window = signal.windows.tukey(256) # ventana de Tukey de 256 muestras

f, t, Sxx = signal.spectrogram(cuatro_latidos, fs, window)
plt.pcolormesh(t,f,Sxx)
plt.title("Espectrograma de 4 latidos, con ventana de 256 muestras")
plt.ylabel('Frecuencia [Hz]')
plt.xlabel('Tiempo [sec]')
plt.show()


plt.figure(2)
window = signal.windows.tukey(3000) # ventana de Tukey de 256 muestras

f, t, Sxx = signal.spectrogram(ecg, fs, window)
plt.pcolormesh(t,f,np.log10(Sxx))
plt.title("Espectrograma de toda la derivacion, con ventana de 3000 muestras")
plt.ylabel('Frecuencia [Hz]')
plt.xlabel('Tiempo [sec]')
plt.show()

