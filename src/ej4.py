'''

Ruidos estacionarios:
	60hz del suministro de energia (ver espectrograma)

Ruidos no estacionarios:
	// Ruido (dinamico?) natural
	Ruido al final de la senial, flasho la medicion

'''

from ecg import *


# for m in marcas[:1000]:
# 	plt.axvline(x=m, color='red')


fig, ax = plt.subplots()
tiempo = np.linspace(0, len(ecg)/200, len(ecg))
plt.plot(tiempo,ecg)
#plt.axis([100/200, 800/200, -1, 1.5])

plt.axvspan(1080, 1650, color='r', alpha=0.45, lw=0)


plt.title("Derivacion, destacando en rojo el ruido no estacionario")
plt.ylabel("Milivolt [mV]")
plt.xlabel("Tiempo [s]")


plt.show()
