
from scipy import signal
from scipy.fftpack import fft
from scipy.fftpack import fftshift
import scipy.io as scp
import matplotlib.pyplot as plt
import numpy as np
#from qrs_detectionok import qrs_detection
from qrs_detectionok import qrs_detection
from picos import detect_peaks


#from ej1-3 import transformada_fourier

SMALL=8
NORMAL=10
BIG=12
BIGG=14

def leer_datos (ts):

    datamat = scp.loadmat('103m.mat')

    datamat_y = datamat['y']
    y = np.array(datamat_y)
    y = y[0,:]

    t = np.arange(361112) ## quiero hacer sizeof... 361112

    t = t * ts * 1000  ##  vector de tiempo en ms

    return t,y

def rta_freq (num, den, nombre, archivo, carpeta):

    [w, H] = signal.freqz(num, den)
    plt.subplot(2,1,1)
    plt.plot(w / np.pi, 20 * np.log10(np.abs(H)))
    plt.xticks(np.arange(0,1.1,0.1))
    plt.ylabel('Modulo [dB]')
    plt.grid(b=1, which='both', axis='both', linewidth=0.5)
    plt.title('Respuesta en frecuencia de %s'%nombre)

    plt.subplot(2,1,2)
    plt.plot(w / np.pi, np.angle(H) / np.pi)
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.ylabel(r'Fase $[\frac{rad}{\pi}]$') # fase normalizada por sobre pi 1---pi lo mismo en frecuencia
    plt.grid(b=1, which='both', axis='both', linewidth=0.5)
    plt.xlabel(r'Frecuencia normalizada $[\frac{\Omega}{\pi}]$')
    #plt.show()
    plt.rc('axes', titlesize=14)  # fontsize of the axes title
    plt.rc('axes', labelsize=14)  # fontsize of the x and y labels
    plt.savefig('imagenes/%s/%s.jpg'%(carpeta,archivo),dpi=300)
    plt.close()


def diag_pyz (num,den, nombre, archivo,carpeta):
    z = np.roots(num)
    p = np.roots(den)
    plt.plot(np.real(z), np.imag(z), 'ob', markerfacecolor='None',markersize=10)
    plt.plot(np.real(p), np.imag(p), 'xr',markersize=10)
    plt.grid(which='both')
    plt.xlabel("Re(Z)")
    plt.ylabel("Im(Z)")
    plt.plot(np.cos(np.r_[0:6.5:0.1]), np.sin(np.r_[0:6.5:0.1]), color='0.2')  # circulo unitario
    plt.title('Polos y ceros de %s'%nombre)
    plt.rc('axes', titlesize=12)  # fontsize of the axes title
    plt.rc('axes', labelsize=12)  # fontsize of the x and y labels
    plt.savefig('imagenes/%s/%s.jpg'%(carpeta,archivo),dpi=300)
    #plt.show()
    plt.close()

"""Recibe una transferencia definida por num y den, el nombre del grafico, el nombre del archivo, la cantidad de puntos a procesar, la carpeta donde se guarda y la mini
ma division de grilla en el eje x"""
def rta_impulso (num, den, nombre, archivo, cant, carpeta, minx):
    x_in = np.zeros(cant+1);
    x_in[0] = 1
    y_out = signal.lfilter(num, den, x_in)
    n = np.r_[0:cant+1]

    fig = plt.figure(1)

    ax = fig.add_subplot(1, 1, 1)

    # Major ticks every 20, minor ticks every 5
    x_major_ticks = np.arange(0,cant+1,2)
    # x_minor_ticks = np.arange(0, 21, 1)
    y_major_ticks = np.arange(0,y_out.max() + 1,minx)

    #y_minor_ticks = np.arange(0,1,0.2)

    ax.set_xticks(x_major_ticks)
    # ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(y_major_ticks)
    #ax.set_yticks(y_minor_ticks, minor=True)

    # And a corresponding grid
    ax.grid(which='both')


    plt.stem(n, y_out, 'r')
    plt.rc('axes', titlesize=14)  # fontsize of the axes title
    plt.rc('axes', labelsize=14)  # fontsize of the x and y labels
    plt.title('Respuesta al impulso de %s'%nombre)
    #plt.xlim(-1, 21)
    plt.xlim([-1 ,cant+2])
    plt.xlabel('n')
    plt.ylabel('h (n)')
    #plt.show()
    plt.savefig('imagenes/%s/%s.jpg'%(carpeta,archivo),dpi=300)
    plt.close()

def rta_escalon(num, den, nombre, archivo,cantidad,carpeta):

    x_in=np.ones(cantidad)
    y_out = signal.lfilter(num, den ,x_in)
    n = np.r_[0:cantidad]

    fig = plt.figure(1)

    ax = fig.add_subplot(1, 1, 1)

    # Major ticks every 20, minor ticks every 5
    x_major_ticks = np.arange(-1, cantidad+1, 2)
    #x_minor_ticks = np.arange(0, 21, 1)
    y_major_ticks = np.arange(0,41,2)
    #y_minor_ticks = np.arange(0,41,1)

    ax.set_xticks(x_major_ticks)
    #ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(y_major_ticks)
    #ax.set_yticks(minor_ticks, minor=True)

    # And a corresponding grid
    ax.grid(which='both')


    plt.stem(n, y_out, 'r')
    plt.xlabel('n')
    plt.ylabel('s (n)')
    plt.title('Respuesta al escalon de %s'%nombre)

    plt.savefig('imagenes/%s/%s.jpg'%(carpeta,archivo),dpi=300)
    plt.close()

def filter_signal (num,den,entrada):

    y = signal.lfilter(num,den,entrada)

    return y

def compensar_retardo(N,entrada):

    ceros = np.zeros(N)
    salida = np.concatenate((entrada[N:len(entrada)], ceros))  # CORRO A LA IZQUIERDA N
    return salida

def transformada_fourier (y, fs, n, nombre, ymin, ymax):
    plt.figure()

    A = fft(y,n)

    freq = np.linspace(-fs/2, fs/2, len(A)) # -fs/2 fs/2 y dsp pueden hacer xlim ([0, fs/2])

    response = 20 * np.log10(np.abs(fftshift(A)))

    plt.plot(freq, response)

    #plt.axis([-100, 100, ymin, ymax])
    plt.xlim([0,fs/2])
    plt.title("Carácterísticas en frecuencia de complejos QRS")
    plt.ylabel("Modulo [dB]")
    plt.xlabel("Frecuencia [Hz]")
    plt.xticks(np.arange(0, 101, 5))
    plt.yticks(np.arange(-100,75,5))
    plt.axis([0, 100, ymin, ymax])
    plt.grid(which='both')
    #plt.show()
    plt.rc('axes', titlesize=12)  # fontsize of the axes title
    plt.rc('axes', labelsize=12)  # fontsize of the x and y labels
    plt.savefig('imagenes/ej5/ej56_%s.jpg'%nombre,dpi=300)
    plt.close()

def evaluar_performance(marcas_propias,marcas_dadas,tolerancia):

    acertados = 0

    for i in range (0,len(marcas_dadas)):
        for j in range (0,len(marcas_propias)):
            if np.abs(marcas_dadas[i]-marcas_propias[j]) <= tolerancia:
                acertados += 1


    falsos_positivos = len(marcas_propias) - acertados
    falsos_negativos = len(marcas_dadas) - acertados


    return acertados, falsos_positivos, falsos_negativos

def main():

    """EJERCICIO 5:"""

    """DEFINO HL (W)"""
    num_hl = np.zeros(13)
    den_hl = np.zeros(13)

    num_hl[0] = 1
    num_hl[6] = -2
    num_hl[12] = 1  # MENOS SIGN

    den_hl[0] = 1
    den_hl[1] = -2
    den_hl[2] = 1

    #rta_freq(num_hl, den_hl, 'ROEUBA', 'prueba', 'ej5/HL')
    """DIAG POLOS Y ZEROS HL"""
    diag_pyz(num_hl, den_hl,'HL','ej5_diag_pyz_hl','ej5/HL')

    """RTA EN FREQ HL"""
    rta_freq(num_hl,den_hl,'HL','ej5_rta_freq_hl','ej5/HL')

    """RTA AL IMPULSO HL"""
    rta_impulso(num_hl, den_hl,'HL','ej5_rta_impulso_hl',20,'ej5/HL',1)

    """RTA AL ESCALON HL"""
    rta_escalon(num_hl, den_hl, 'HL', 'ej5_rta_escalon_hl',20,'ej5/HL')

    """RETARDO DE HL"""

    w, gd = signal.group_delay((num_hl,den_hl))
    plt.plot(w/np.pi,gd)
    plt.title("Retardo producido por HL")
    plt.xlabel ('Pulsacion normalizada/pi ')
    plt.ylabel('Retardo en cantidad de muestras')
    #plt.show()
    plt.rc('axes', titlesize=12)  # fontsize of the axes title
    plt.rc('axes', labelsize=12)  # fontsize of the x and y labels
    plt.savefig('imagenes/ej5/g_delay_hl.jpg',dpi=300)
    plt.close()

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    """DEFINO HH (W)"""
    num_hh = np.zeros(33, dtype=float)
    den_hh = np.zeros(33, dtype=float)

    num_hh[0] = -1/32
    num_hh[16] = 1
    num_hh[17] = -1
    num_hh[32] = 1/32 # MENOS SIGNIFICATIVO

    den_hh[0] = 1
    den_hh[1] = -1

    """DIAG POLOS Y ZEROS HH"""
    diag_pyz(num_hh, den_hh,'HH','ej5_diag_pyz_hh','ej5/HH')

    """RTA EN FREQ HH"""
    rta_freq(num_hh,den_hh,'HH','ej5_rta_freq_hh','ej5/HH')

    """RTA AL IMPULSO HH"""
    rta_impulso(num_hh,den_hh,'HH','ej5_rta_impulso_hh',30,'ej5/HH',0.2)

    """RTA AL ESCALON HH"""
    rta_escalon(num_hh, den_hh, 'HH', 'ej5_rta_escalon_hh',40,'ej5/HH')

    """RETARDO DE HH"""

    w, gd = signal.group_delay((num_hh,den_hh)) # 16 muestras
    plt.plot(w/np.pi,gd)
    plt.grid(which='both')
    plt.title("Retardo producido por HH")
    plt.xlabel('Pulsacion normalizada/pi ')
    plt.ylabel('Retardo en cantidad de muestras')
    plt.rc('axes', titlesize=12)  # fontsize of the axes title
    plt.rc('axes', labelsize=12)  # fontsize of the x and y labels
    plt.savefig('imagenes/ej5/g_delay_hh.jpg', dpi=300)
    #plt.show()

    plt.close()


    """"""""""""""""""""""""""""""""""""""""""""""""

    """EFECTO SOBRE LA SENIAL"""
    fs = 200;
    L = leer_datos(1/fs)
    t = L[0]
    y = L[1]
    y = (y - y.mean()) / 200 # ganancia 200
    #y = y/200
    #y_out_hl = signal.lfilter(num_hl, den_hl, y)
    #y_out_hh   = signal.lfilter(num_hh,den_hh,y_out_hl)
    y_out_hl = filter_signal(num_hl,den_hl,y)
    y_out_hl = compensar_retardo(5,y_out_hl)
    y_out_hh = filter_signal(num_hh,den_hh,y_out_hl)
    y_out_hh = compensar_retardo(16,y_out_hh)

    fig=plt.figure()
    ax1=fig.add_subplot(3,1,1)
    plt.plot(t, y)
    plt.title('Señal original')
    plt.xticks(fontsize=0)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.yticks(np.arange(-1,1.5,0.5))
    plt.grid(which='both')
    #plt.xlabel('Tiempo [ms]')
    #plt.ylabel('Amplitud [mV]')

    plt.xlim(2000,4000)
    plt.ylim(-1,1.5)
    #plt.show()
    #plt.savefig('imagenes/ej5/y_original.jpg',dpi=300)
    #plt.close()



    ax2=fig.add_subplot(3,1,2)
    plt.plot(t,y_out_hl)
    plt.title ('Señal filtrada con HL')
    plt.setp(ax2.get_xticklabels(), visible=False)
    #plt.xticks(np.arange(1999, 4000, 2002))
    plt.xticks(fontsize=0)
    plt.yticks(np.arange(-20, 25, 5))
    plt.grid(which='both')
    #plt.xlabel('Tiempo [ms]')

    plt.ylabel('Amplitud [mV]')
    plt.xlim(2000, 4000)
    plt.ylim(-15, 20)
    #plt.show()
    #plt.savefig('imagenes/ej5/y_filtradahl.jpg',dpi=300)
    #plt.close()


    """"""""""""

    plt.subplot(3,1,3)
    plt.plot(t, y_out_hh)
    plt.title('Señal filtrada con HH')
    plt.xlabel('Tiempo [ms]')
    #abel('Amplitud [mV]')
    plt.yticks(np.arange(-20, 25, 5))
    plt.grid(which='both')
    plt.xlim(2000, 4000)
    plt.ylim(-15, 22)
    #plt.show()
    plt.rc('axes', titlesize=12)  # fontsize of the axes title
    plt.rc('axes', labelsize=12)  # fontsize of the x and y labels
    plt.savefig('imagenes/ej5/y_filtradatodos.jpg',dpi=300)
    plt.close()

    """TRANSFORMADAS DE CADA UNO"""

    #transformada_fourier(y[137:160],fs,1024,"y_original",-65,20)
    #transformada_fourier(y_out_hl[137:160],fs,1024,"y_hl",-20,50)
    #transformada_fourier(y_out_hh[137:160],fs,1024,"y_hh",-20,50)

    fig = plt.figure()
    n = 1024
    fs = 200

    A = fft(y[137:160],n)
    B = fft(y_out_hl[137:160],n)
    C = fft(y_out_hh[137:160],n)

    freq = np.linspace(-fs/2, fs/2, len(A)) # -fs/2 fs/2 y dsp pueden hacer xlim ([0, fs/2])

    response = 20 * np.log10(np.abs(fftshift(A)))
    response_hl = 20 * np.log10(np.abs(fftshift(B)))
    response_hh = 20 * np.log10(np.abs(fftshift(C)))

    ax1=fig.add_subplot(3,1,1)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.plot(freq, response)
    #plt.axis([-100, 100, ymin, ymax])
    plt.xlim([0,fs/2])
    plt.title("Carácterísticas en frecuencia QSR:Filtrado pasa-banda")
    #plt.ylabel("Modulo [dB]")
    #plt.xlabel("Frecuencia [Hz]")
    plt.xticks(np.arange(0, 101, 5))
    plt.yticks(np.arange(-100,80,10))
    plt.axis([0, 100, -65, 20])
    plt.grid(which='both')
    #plt.show()
    #plt.savefig('imagenes/ej5/ej56_y_original.jpg',dpi=300)
    #plt.close()

    ax2 = fig.add_subplot(3, 1, 2)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.plot(freq, response_hl)
    # plt.axis([-100, 100, ymin, ymax])
    plt.xlim([0, fs / 2])
    #plt.title("Carácterísticas en frecuencia QSR_HL")
    plt.ylabel("Modulo [dB]")
    # plt.xlabel("Frecuencia [Hz]")
    plt.xticks(np.arange(0, 101, 5))
    plt.yticks(np.arange(-100, 80, 10))
    plt.axis([0, 100, -20, 50])
    plt.grid(which='both')

    ax3 = fig.add_subplot(3, 1, 3)
    #plt.setp(ax2.get_xticklabels(), visible=False)
    plt.plot(freq, response_hh)
    # plt.axis([-100, 100, ymin, ymax])
    plt.xlim([0, fs / 2])
    #plt.title("Carácterísticas en frecuencia QSR_HH")
    # plt.ylabel("Modulo [dB]")
    plt.xlabel("Frecuencia [Hz]")
    plt.xticks(np.arange(0, 101, 5))
    plt.yticks(np.arange(-100, 80, 10))
    plt.axis([0, 100, -20, 50])
    plt.grid(which='both')
    plt.rc('axes', titlesize=12)  # fontsize of the axes title
    plt.rc('axes', labelsize=12)  # fontsize of the x and y labels
    plt.savefig('imagenes/ej5/ej56_y_filtrados_freq.jpg',dpi=300)
    plt.close()

    """EJERCICIO 8"""

    time_sample = 1/fs
    h = 1

    num_hd = np.zeros(5)
    den_hd = 1

    num_hd[0] = 1 / (8*h)
    num_hd[1] = 1 / (4*h)
    num_hd[3] = -1 / (4 * h)
    num_hd[4] = -1 / (8*h) #MENOS SIGNIFICATIVO

    rta_freq(num_hd,den_hd,'HD','ej8rta_freq_hd','ej8/HD')


    y_out_hd = filter_signal(num_hd,den_hd,y_out_hh)

    """LINEAL DE HD"""
    [w, H] = signal.freqz(num_hd, den_hd)
    plt.subplot(2, 1, 1)
    plt.plot(w / np.pi, np.abs(H))
    plt.ylabel('Modulo [dB]')
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.grid(b=1, which='both', axis='both', linewidth=0.5)
    plt.title('Respuesta en frecuencia de HD LINEAL')

    plt.subplot(2, 1, 2)
    plt.plot(w / np.pi, np.angle(H) / np.pi)
    plt.ylabel('Fase [rad/pi]')  # fase normalizada por sobre pi 1---pi lo mismo en frecuencia
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.grid(b=1, which='both', axis='both', linewidth=0.5)
    plt.xlabel('Frecuencia normalizada x/pi')

    # plt.show()
    plt.rc('axes', titlesize=12)  # fontsize of the axes title
    plt.rc('axes', labelsize=12)  # fontsize of the x and y labels
    plt.savefig('imagenes/ej8/hd_rta_lineal.jpg' , dpi=300)
    plt.close()



    #plt.figure()
    #plt.plot(t, y_out_hd)

    #print(y_out_hd)
    y_out_hd = compensar_retardo(2,y_out_hd)
    #y_out_hd = np.concatenate((y_out_hd[2:len(y_out_hd)], ceros)) #CORRO A LA IZQUIERDA DOS
    # ceros = np.zeros(2)

    plt.figure()
    #print(y_out_hd)
    #plt.subplot()
    plt.plot(t, y_out_hd)
    plt.title('Señal filtrada con HD')
    plt.xlabel('Tiempo [ms]')
    plt.ylabel('Amplitud [mV]')
    plt.xlim(2000, 4000)
    plt.ylim(-6, 6)
    plt.grid(which='both')
    # plt.show()
    plt.rc('axes', titlesize=12)  # fontsize of the axes title
    plt.rc('axes', labelsize=12)  # fontsize of the x and y labels
    plt.savefig('imagenes/ej8/y_filtradahd.jpg', dpi=300)
    plt.close()




    """EJERCICIO 9"""

    y_out_cuad = np.power (y_out_hd,2)


    plt.figure()
    plt.plot(t,y_out_cuad)
    plt.title('Señal al cuadrado')
    plt.xlabel('Tiempo [ms]')
    plt.ylabel('Amplitud [mV]')
    plt.xlim(2000, 4000)
    plt.ylim(-1, 35)
    plt.grid(which='both')
    # plt.show()
    plt.rc('axes', titlesize=12)  # fontsize of the axes title
    plt.rc('axes', labelsize=12)  # fontsize of the x and y labels
    plt.savefig('imagenes/ej9/y_cuad.jpg', dpi=300)
    plt.close()

    """EJERCICIO 10""" ## 1/N *(1-z^-N)/(1-z^-1)

    N = 20 #PROBAR 20
    
    num_hi = np.zeros(N+1)
    den_hi = np.zeros (2)

    num_hi[0] = 1
    num_hi[N] = -1

    den_hi[0] = N
    den_hi[1] = -N

    rta_freq(num_hi,den_hi,'HI','ej10rta_freq_hi','ej10/HI') # GROUP DELAY DE 15 MUESTRAS


    y_out_hi = filter_signal(num_hi,den_hi,y_out_cuad)

    #ceros = np.zeros(15)
    #y_out_hi = np.concatenate((y_out_hi[15:len(y_out_hi)], ceros))  # CORRO A LA IZQUIERDA QUINCE
    y_out_hi = compensar_retardo(int (N/2), y_out_hi)

    plt.plot(t, y_out_hi)
    plt.title('Señal integrada')
    plt.xlabel('Tiempo [ms]')
    plt.ylabel('Amplitud [mV]')
    plt.xlim(2000, 4000)
    plt.ylim(-1, 10)
    plt.grid(which='both')
    # plt.show()
    plt.rc('axes', titlesize=12)  # fontsize of the axes title
    plt.rc('axes', labelsize=12)  # fontsize of the x and y labels
    plt.savefig('imagenes/ej10/y_filtradahi.jpg', dpi=300)
    plt.close()

    plt.plot(t, y_out_hi)
    plt.title('Señal integrada')
    plt.xlabel('Tiempo [ms]')
    plt.ylabel('Amplitud [mV]')
    plt.xlim(2000, 4000)
    plt.ylim(-1, 20)
    plt.grid(which='both')
    # plt.show()
    plt.subplot()
    plt.plot(t, y_out_cuad,'--r') # PARA VER SUPERPUESTA EL CUAD
    plt.legend(["Señal integrada","Señal al cuadrado"])
    plt.rc('axes', titlesize=12)  # fontsize of the axes title
    plt.rc('axes', labelsize=12)  # fontsize of the x and y labels
    plt.savefig('imagenes/ej10/y_filtradahi_super.jpg', dpi=300)
    plt.close()

    """EJERCICIO 12"""

    peaks_index, diccionario = signal.find_peaks(y_out_hi,height=0,distance=40)
    peaks_altura = diccionario['peak_heights']

    peaks_absolutos = np.zeros(len(y_out_hi))
    n = 0  #CONTADOR

    for i in range (0,len(y_out_hi)):
        if i in peaks_index:
            peaks_absolutos[i] = peaks_altura[n]
            n += 1

    peaks_final = qrs_detection(peaks_absolutos)

    """plt.figure()
    plt.plot(range(0, len(peaks_final)), peaks_final, 'ro')
    plt.subplot()

    plt.plot(t * fs / 1000, y)
    plt.show()"""

    """#convierto en NULL los 0 para plotear lindo

    for i in range (0,len(peaks_final)):
        if peaks_final[i]!=0:
            peaks_final[i]=1
        else:
            peaks_final[i]= 10000

    plt.figure()

    plt.plot(range(0, len(peaks_final)), peaks_final, 'ro')
    plt.subplot()
    #plt.ylim(-1,1.5)
    plt.plot(t * fs / 1000, y)
    plt.show()
"""


    """EFICIENCIA"""
    marcas_dadas = np.loadtxt(fname='marcas.txt')


    marcas_propias = np.array([])
    for i in range(0,len(peaks_final)):
        if peaks_final[i] != 0:
            marcas_propias = np.concatenate( (marcas_propias,np.array([i])))

    acertados,falsos_positivos,falsos_negativos = evaluar_performance(marcas_propias,marcas_dadas,10)

    print ("acertados = %d, falsos_positivos = %d, falsos_negativos = %d"%(acertados,falsos_positivos,falsos_negativos))
    efect = 100 * acertados/len(marcas_dadas)
    print ("efectividad = %.2f"%efect)


main()