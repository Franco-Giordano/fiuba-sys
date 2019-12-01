from preprocesar import preprocesar
from ecg import *
import math
from ej5 import filtrar


def decimar_y_expandir(entrada, puntos_sinc=2000):

	CANT_SOBREMUESTREO = 9

	CANT_DECIMACION = 5

	sobremuestreada = []

	cantidad_retraso = puntos_sinc//2

	for e in entrada:
		sobremuestreada += [e]
		sobremuestreada += [0 for i in range(CANT_SOBREMUESTREO-1)]

	x = np.arange(-(puntos_sinc-1)/2, (puntos_sinc-1)/2)

	h_lp = np.sinc(x / CANT_SOBREMUESTREO)


	salida_filtro = np.convolve(sobremuestreada, h_lp, mode='full')

	salida_filtro = np.concatenate((salida_filtro, np.zeros(cantidad_retraso)))
	salida_filtro = salida_filtro[cantidad_retraso:]

	decimar = salida_filtro[::5]

	return decimar


# rta imp decimar+expandir

imp = [0 for i in range(len(ecg_original))]
imp[0] = 1

rta_imp = decimar_y_expandir(imp)

fft = fft(rta_imp,2048)  # fft de 2048 puntos
freq = np.linspace(-math.pi, math.pi, len(fft))
response = 20 * np.log10(np.abs(fftshift(fft/abs(fft).max())))

plt.title("Respuesta en frec")
plt.xlabel("Ω")
plt.ylabel("Amplitud [dB]")
plt.xlim(0,math.pi)
plt.plot(freq,response)

plt.show()

ecg_remuestreado = decimar_y_expandir(ecg_original) / 360

# ============== HL =================

b,a = [0 for i in range(13)], [0 for i in range(3)]

b[0] = 1
b[6] = -2
b[12] = 1

a[0] = 1
a[1] = -2
a[2] = 1

impulso = np.zeros(36); impulso[0] = 1
rta_imp = filtrar(b,a,impulso, title='Respuesta al impulso de HL(z)', plot_results=False, cantidad_retraso=0)

rta_imp_remuestreada = decimar_y_expandir(rta_imp)

salida_hl = np.convolve(ecg_remuestreado, rta_imp_remuestreada)

cantidad_retraso = 9 # 5 * 9/5

salida_hl = np.concatenate((salida_hl, np.zeros(cantidad_retraso)))
salida_hl = salida_hl[cantidad_retraso:]

plt.title("ECG vs HL")
plotear(ecg_remuestreado[100:800], show=False)
plotear(salida_hl[100:800])


# ==================== HH =======================

b,a = [0 for i in range(33)], [0 for i in range(2)]

b[0] = -1/32
b[16] = 1
b[17] = -1
b[32] = 1/32

a[0] = 1
a[1] = -1

impulso = np.zeros(36); impulso[0] = 1

rta_imp = filtrar(b,a,impulso, plot_results=False, cantidad_retraso=0)

rta_imp_remuestreada = decimar_y_expandir(rta_imp)

salida_hh = np.convolve(salida_hl, rta_imp_remuestreada)

cantidad_retraso = 29 # 16 * 9/5 = 28.8

salida_hh = np.concatenate((salida_hh, np.zeros(cantidad_retraso)))
salida_hh = salida_hh[cantidad_retraso:]

plt.title("HL vs HH")
plotear(salida_hl[100:800], show=False)
plotear(salida_hh[100:800])

# ==================== DERIVADOR ======================

coef_num = [1,2,0,-2,-1]

coef_den = [8]

impulso = np.zeros(36); impulso[0] = 1

rta_imp = filtrar(coef_num, coef_den, impulso, plot_results=False, cantidad_retraso=0)

salida_hd = np.convolve(salida_hh, rta_imp)

cantidad_retraso = 2 # 

salida_hd = np.concatenate((salida_hd, np.zeros(cantidad_retraso)))
salida_hd = salida_hd[cantidad_retraso:]



plt.title("HH vs HD")
plotear(salida_hh[100:800], show=False)
plotear(salida_hd[100:800])

