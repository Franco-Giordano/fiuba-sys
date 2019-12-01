from ecg import *

from ej5 import filtrar

def preprocesar(senial, dbs_ruido=None):

	senial_entrada = np.array(senial)

	if dbs_ruido is not None:

		desvio = np.sqrt(np.var(senial_entrada) / (10**(dbs_ruido/10)))

		ruido = np.random.normal(scale=desvio, size=len(senial_entrada))

		senial_entrada += ruido

	b,a = [0 for i in range(13)], [0 for i in range(3)]

	b[0] = 1
	b[6] = -2
	b[12] = 1

	a[0] = 1
	a[1] = -2
	a[2] = 1

	salida_hl = filtrar(b,a,senial_entrada,title="Señal ECG filtrada por HL(z)", usar_plot=True, intervalo=[100,800], cantidad_retraso=5,  plot_results=False)

	b,a = [0 for i in range(33)], [0 for i in range(2)]

	b[0] = -1/32
	b[16] = 1 
	b[17] = -1
	b[32] = 1/32

	a[0] = 1
	a[1] = -1

	salida_hh = filtrar(b,a,salida_hl, title="Señal de salida de HH(z), con entrada HL(z).ECG(z)", usar_plot=True, intervalo=[100,800], cantidad_retraso=16, plot_results=False)

	coef_num = [1,2,0,-2,-1]

	coef_den = [8]

	# ES DOS DE CORRECCION YA QUE PENDIENTE 2.14 Y LO RETRASE 2 MUESTRAS PARA QUE SEA CAUSAL
	salida_derivada = filtrar(coef_num, coef_den, salida_hh, title="Derivada de la señal filtrada", intervalo=[100,800], usar_plot=True, cantidad_retraso=2,  plot_results=False)

	senial_al_cuadrado = np.power(salida_derivada, 2)

	N = 20

	coef_num = np.ones(N)
	coef_den = [N]

	integrada = filtrar(coef_num, coef_den, senial_al_cuadrado, title="Integrada de la señal derivada", intervalo=[100,800], usar_plot=True, cantidad_retraso=N//2,  plot_results=False)

	return integrada

