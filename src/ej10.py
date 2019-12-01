from ecg import *
from ej5 import rta_frecuencia, filtrar

# h(n) = ( sum [delta(n-k)] de k=0 a N-1 ) / N
# => H(z) = (1 + z^-1 + z^-2 + ... + z^N-1) / N

al_cuadrado = np.loadtxt('ecg_filtrado_hl_hh_hd_cuadrado.out')

# para ver el largo de un QRS ('rtaECGfiltrado-cuadrado-enfoque.png')
# plotear(al_cuadrado)

# viendo el grafico de arriba, se ve que cada QRS es aproximadamente de 20 muestras
N = 20

coef_num = np.ones(N)
coef_den = [N]

rta_frecuencia(coef_num, coef_den, title='Respuesta en frecuencia del integrador')

integrada = filtrar(coef_num, coef_den, al_cuadrado, title="Integrada de la señal derivada", intervalo=[100,800], usar_plot=True, cantidad_retraso=N//2)


# integrada = np.concatenate((integrada, resto_integrada)) / N

np.savetxt("ecg_filtrado_hl_hh_hd_cuadrado_integrada.out", integrada)