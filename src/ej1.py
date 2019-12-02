from ecg import *


# for m in marcas[:1000]:
# 	plt.axvline(x=m, color='red')

print(ecg)


fig, ax = plt.subplots()
tiempo = np.linspace(0, len(ecg)/200, len(ecg))
plt.plot(tiempo,ecg)
plt.axis([100/200, 800/200, -1, 1.5])

ax.axvline(x=294/200, ymin=0.0, ymax=1.0, color='r', ls=':')	#P

ax.axvline(x=314/200, ymin=0.0, ymax=1.0, color='r', ls=':')	#Q

ax.axvline(x=320/200, ymin=0.0, ymax=1.0, color='r', ls=':')	#R

ax.axvline(x=324/200, ymin=0.0, ymax=1.0, color='r', ls=':')	#S

ax.axvline(x=369/200, ymin=0.0, ymax=1.0, color='r', ls=':')	#T

ax.axvline(x=410/200, ymin=0.0, ymax=1.0, color='r', ls=':')	#U





plt.title("Un ciclo QRS")
plt.ylabel("Milivolt [mV]")
plt.xlabel("Tiempo [s]")


plt.show()
