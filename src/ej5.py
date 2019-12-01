from ecg import *



def rta_frecuencia(num,den, title="BODE"):

	[w,H] = signal.freqz(num,den)
	plt.subplot(211), plt.plot(w/np.pi,20*np.log10(np.abs(H))), plt.ylabel('Modulo [dB]'), plt.grid()
	plt.title(title)
	plt.subplot(212),plt.plot(w/np.pi,np.angle(H)),plt.ylabel('Fase [rad]'),plt.grid()
	plt.xlabel('Frecuencia normalizada x/pi')
	plt.show()

def filtrar(num, den, entrada, title="SALIDA", usar_plot=False, intervalo=None, cantidad_retraso=0, plot_results=True):
	respuesta = signal.lfilter(num,den,entrada)
	tiempo = np.r_[0:len(entrada)]

	plot_x = tiempo
	plot_y = respuesta

	if cantidad_retraso:
		respuesta = np.concatenate((respuesta, np.zeros(cantidad_retraso)))
		respuesta = respuesta[cantidad_retraso:]

	if plot_results:
		if intervalo is not None:
			inicio = intervalo[0]
			fin = intervalo[1]
			plot_y = respuesta[inicio:fin]
			plot_x = tiempo[inicio:fin]
		
		if not usar_plot:
			plt.stem(plot_x,plot_y,'r')
		else:
			plt.plot(plot_x, plot_y)

		plt.title(title)
		
		plt.show()
	return respuesta


if __name__ == '__main__':

	b,a = [0 for i in range(13)], [0 for i in range(3)]

	b[0] = 1
	b[6] = -2
	b[12] = 1

	a[0] = 1
	a[1] = -2
	a[2] = 1


	num = np.roots(b)
	den = np.roots(a)

	plt.plot(np.cos(np.r_[0:6.5:0.1]),np.sin(np.r_[0:6.5:0.1]),color='0.7') # circulo unitario
	plt.plot(np.real(num),np.imag(num),'ob',markerfacecolor='None', markersize=6)
	plt.plot(np.real(den),np.imag(den),'Xb',markersize=10, color='red'),plt.grid()
	plt.title('Polos y ceros de HL(z)'),plt.show()

	rta_frecuencia(b,a, title='Respuesta en frecuencia de HL(z)')


	impulso = np.zeros(36); impulso[0] = 1
	filtrar(b,a,impulso, title='Respuesta al impulso de HL(z)')

	salida_hl = filtrar(b,a,ecg,title="Señal ECG filtrada por HL(z)", usar_plot=True, intervalo=[100,800], cantidad_retraso=5)


	b,a = [0 for i in range(33)], [0 for i in range(2)]

	b[0] = -1/32
	b[16] = 1 
	b[17] = -1
	b[32] = 1/32

	a[0] = 1
	a[1] = -1


	num = np.roots(b)
	den = np.roots(a)

	plt.plot(np.cos(np.r_[0:6.5:0.1]),np.sin(np.r_[0:6.5:0.1]),color='0.7') # circulo unitario
	plt.plot(np.real(num),np.imag(num),'ob',markerfacecolor='None', markersize=6)
	plt.plot(np.real(den),np.imag(den),'Xb',markersize=10, color='red'),plt.grid()
	plt.title('Polos y ceros de HH(z)'),plt.show()

	rta_frecuencia(b,a, title='Respuesta en frecuencia de HH(z)')
	plt.xlabel("Numero de muestra")
	plt.ylabel("Amplitud [mV]")
	filtrar(b,a,impulso,title='Respuesta al impulso de HH(z)')

	salida_filtrada = filtrar(b,a,salida_hl, title="Señal de salida de HH(z), con entrada HL(z).ECG(z)", usar_plot=True, intervalo=[100,800], cantidad_retraso=16)

	np.savetxt("ecg_filtrado_hl_hh.out",salida_filtrada)
