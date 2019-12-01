from ecg import *

from ej5 import rta_frecuencia, filtrar


if __name__ == '__main__':
	coef_num = [1,2,0,-2,-1]

	coef_den = [8]

	entrada = np.loadtxt("ecg_filtrado_hl_hh.out")

	rta_frecuencia(coef_num, coef_den, title="Respuesta en frecuencia de HD(z)")

	# ES DOS DE CORRECCION YA QUE PENDIENTE 2.14 Y LO RETRASE 2 MUESTRAS PARA QUE SEA CAUSAL
	salida_derivada = filtrar(coef_num, coef_den, entrada, title="Derivada de la señal filtrada", intervalo=[100,800], usar_plot=True, cantidad_retraso=2)

	np.savetxt("ecg_filtrado_hl_hh_hd.out", salida_derivada)

