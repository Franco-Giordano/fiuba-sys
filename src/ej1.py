from ecg import *


print(len(marcas), marcas)

delta = [marcas[i] - marcas[i-1] for i in range(1, len(marcas[:1000]))]
print("DELTA PROM:",sum(delta)/len(delta))

# for m in marcas[:1000]:
# 	plt.axvline(x=m, color='red')

print(ecg)


fig, ax = plt.subplots()
plt.plot(ecg)

ax.axvline(x=294, ymin=0.0, ymax=1.0, color='r', ls=':')	#P

ax.axvline(x=314, ymin=0.0, ymax=1.0, color='r', ls=':')	#Q

ax.axvline(x=320, ymin=0.0, ymax=1.0, color='r', ls=':')	#R

ax.axvline(x=324, ymin=0.0, ymax=1.0, color='r', ls=':')	#S

ax.axvline(x=369, ymin=0.0, ymax=1.0, color='r', ls=':')	#T

ax.axvline(x=410, ymin=0.0, ymax=1.0, color='r', ls=':')	#U





plt.axis([100, 800, -1, 1.5])
plt.title("Un ciclo caracteristico del ECG")
plt.ylabel("Milivolt [mV]")
plt.xlabel("Numero de muestra")


plt.show()
