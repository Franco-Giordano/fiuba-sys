from ecg import *

from ej5 import rta_frecuencia, filtrar


if __name__ == '__main__':
	coef_num = [1,2,0,-2,-1]

	coef_den = [8]

	entrada = np.loadtxt("ecg_filtrado_hl_hh.out")

	ideal_num = [1,0] # jw
	ideal_den = [0,1] # 1
	[w,H] = signal.freqs(ideal_num,ideal_den)
	plt.plot(w[:84]/np.pi * 100,20*np.log10(np.abs(H[:84])), color='orange', label='Derivador ideal analogico')

	[w,H] = signal.freqz(coef_num,coef_den)
	# plt.subplot(211)
	plt.plot(w/np.pi * 100,20*np.log10(np.abs(H)), label='Derivador no ideal'), plt.ylabel('Modulo [dB]'), plt.grid()
	plt.legend()
	plt.axvline(x=28, color='r', ls=':')
	plt.show()
	#rta_frecuencia(coef_num, coef_den, title="Respuesta en frecuencia de HD(z)")

	# ES DOS DE CORRECCION YA QUE PENDIENTE 2.14 Y LO RETRASE 2 MUESTRAS PARA QUE SEA CAUSAL
	salida_derivada = filtrar(coef_num, coef_den, entrada, title="Derivada de la señal filtrada", intervalo=[100,800], usar_plot=True, cantidad_retraso=2)

	np.savetxt("ecg_filtrado_hl_hh_hd.out", salida_derivada)

