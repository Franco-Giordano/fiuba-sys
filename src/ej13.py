from preprocesar import *
from ej12 import evaluar_performance

marcas_reales = np.loadtxt("../data/marcas.txt")

ecg_con_5 = preprocesar(ecg, dbs_ruido=5)

ecg_con_10 = preprocesar(ecg, dbs_ruido=10)

ecg_con_20 = preprocesar(ecg, dbs_ruido=20)

ecg_con_30 = preprocesar(ecg, dbs_ruido=30)

ecg_con_40 = preprocesar(ecg, dbs_ruido=40)

evaluar_performance(ecg_con_10, marcas_reales, print_header="==== 10 DB ====")

evaluar_performance(ecg_con_20, marcas_reales, print_header="==== 20 DB ====")

evaluar_performance(ecg_con_30, marcas_reales, print_header="==== 30 DB ====")
