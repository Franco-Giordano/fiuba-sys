from ecg import *
from qrs_detection import qrs_detection

senial_final = np.loadtxt("ecg_filtrado_hl_hh_hd_cuadrado_integrada.out")

intensidad_marcas = qrs_detection(senial_final)

marcas_final = [i for i in range(len(intensidad_marcas)) if intensidad_marcas[i] != 0]

plt.figure()
plt.plot(senial_final, color='green')

for i in range(len(intensidad_marcas)):
	if intensidad_marcas[i] != 0:
		plt.axvline(x=i, color='red')
plt.plot(ecg)
plt.show()


plt.figure()

print(len(marcas_final), len(marcas))

#assert(len(marcas_final) == len(marcas))

for i in range(len(marcas)):
	plt.axvline(x=marcas[i], color='green')
	plt.axvline(x=marcas_final[i], color='red')

plotear(ecg)