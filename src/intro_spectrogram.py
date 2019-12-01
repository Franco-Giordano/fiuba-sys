# -*- coding: utf-8 -*-
"""
Ejemplo de espectrograma en Python

Seniales y Sistemas - Curso 1 - FIUBA

Se recomienda ver el help de la funcion ejecutando 
    help(signal.spectrogram)
"""

from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft
from scipy.fftpack import fftshift

'''
Generate a test signal, a 2 Vrms sine wave whose frequency is slowly modulated 
around 3kHz, corrupted by white noise of exponentially decreasing magnitude 
sampled at 10 kHz.
'''
fs = 10e3
N = 1e5
amp = 2*np.sqrt(2)
noise_power = 0.01*fs/2
time = np.arange(N) / float(fs)
mod = 1000*np.cos(2*np.pi*0.25*time) # moduladora (se suma en el argumento de carrier)
carrier = amp*np.sin(2*np.pi*3e3*time + mod)
noise = np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
noise *= np.exp(-time/5)
x = carrier + noiselog10

'''
Compute and plot the spectrogram
'''
window = signal.tukey(256) # ventana de Tukey de 256 muestras
f, t, Sxx = signal.spectrogram(x, fs, window)
plt.pcolormesh(t,f,Sxx)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()

'''
Ploteo en tiempo y frecuencia de la ventana utilizada
'''
plt.plot(window)
plt.title("Tukey window")
plt.ylabel("Amplitude")
plt.xlabel("Sample")
plt.ylim([0, 1.1])

plt.figure()
A = fft(window,2048)/(len(window)/2.0)  # fft de 2048 puntos
freq = np.linspace(-0.5, 0.5, len(A))
response = 20 * np.log10(np.abs(fftshift(A/abs(A).max())))
plt.plot(freq,response)
plt.axis([-0.5, 0.5, -120, 0])
plt.title("Frequency response of the Tukey window")
plt.ylabel("Normalized magnitude [dB]")
plt.xlabel("Normalized frequency [cycles per sample]")

