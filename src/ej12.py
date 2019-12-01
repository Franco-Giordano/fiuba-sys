from ecg import *
from qrs_detection import qrs_detection

def evaluar_performance(senial_entrada, marcas_reales, print_header=""):

	peaks_index, diccionario = signal.find_peaks(senial_entrada,height=0,distance=20)
	peaks_altura = diccionario['peak_heights']

	peaks_absolutos = np.zeros(len(senial_entrada))
	n = 0  #CONTADOR

	for i in range (0,len(senial_entrada)):
	    if i in peaks_index:
	        peaks_absolutos[i] = peaks_altura[n]
	        n += 1

	peaks_final = qrs_detection(peaks_absolutos)

	marcas_mias = [i for i in range(len(peaks_final)) if peaks_final[i] != 0]

	cant_marcas_bien, cant_falsos_negativos = 0, 0

	for m in marcas_reales:
		if np.count_nonzero(peaks_final[int(m)-10:int(m)+10]) > 0: 	# hay al menos una marca dentro del QRS!
			cant_marcas_bien += 1
		else:												# no hay ninguna
			cant_falsos_negativos += 1

		if m < 10 or (m+10) >= len(peaks_final):
			raise ValueError('Accediendo a indices invalidos')

	# DEBERIA CUMPLIRSE QUE:
	# 		len(marcas_reales) = cant_bien + cant_fn
	# 		len(marcas_mias) = cant_bien + cant_fp - cant_fn

	cant_falsos_positivos = len(marcas_mias) - cant_marcas_bien

	# sanity check
	assert(cant_marcas_bien + cant_falsos_negativos == len(marcas_reales))
	assert((len(marcas_reales) + cant_falsos_positivos - cant_falsos_negativos) == len(marcas_mias))

	eficiencia = cant_marcas_bien/len(marcas_reales) * 100

	print(print_header)
	print("Falsos Positivos: {}, Falsos Negativos: {}. Verdaderos Positivos: {}".format(cant_falsos_positivos, cant_falsos_negativos, cant_marcas_bien))
	print("Cantidad esperada: {}. Cantidad obtenida: {}".format(len(marcas_reales), len(marcas_mias)))
	print("Eficiencia: Cant_VP / Cant_Esperada = {}%".format(eficiencia))
	print("\n")

	return marcas_mias, eficiencia

if __name__ == '__main__':
	marcas_reales = np.loadtxt("../data/marcas.txt")

	senial_final = np.loadtxt("ecg_filtrado_hl_hh_hd_cuadrado_integrada.out")

	marcas_mias = evaluar_performance(senial_final, marcas_reales)

	plt.figure()
	plt.plot(senial_final, color='green')

	for i,j in zip(marcas_mias, marcas_reales):
		plt.axvline(x=i, color='red')
		plt.axvline(x=j, color='green')

	plt.plot(ecg)
	plt.show()

