import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
import matplotlib.image as mpimg
from scipy import ndimage
import csv

# EN ESTE PROGRAMA SIMPLEMENTE SE HACE EL TRATAMIENTO DE DATOS Y SE PLOTEAN LOS DATOS


#ruta de los archivos
pathDL = r'C:\Users\836582\Desktop\Brochado\Archivos Brochadora'


teeth_number = 42
first_tooth_position = 915
step = 10
disk_width = 40


numseg = 5
MGnum= 1.27

transition = int(disk_width) / int(step) + numseg

zonaBrochadoInicio = int(first_tooth_position)
zonaEstableInicio = int(first_tooth_position) + (transition * int(step))
zonaEstableFin =  int(first_tooth_position) + (int(teeth_number) - transition) * int(step)
zonaBrochadoFin = int(first_tooth_position) + (int(teeth_number) * int(step))


print("ZONA BROCHADO INICIO: " + str(zonaBrochadoInicio))
print("ZONA ESTABLE INICIO: " + str(zonaEstableInicio))
print("ZONA ESTABLE FIN: " + str(zonaEstableFin))
print("ZONA BROCHADO FIN: " + str(zonaBrochadoFin))

# Donde quiero exportar los errores generados, modelos e imagenes
ruta= r'C:\Users\836582\Desktop\Brochado\export'


colores = ['limegreen', 'teal', 'orange', 'gold', 'navy']
###################### Deactivates warning
pd.options.mode.chained_assignment = None  # default='warn'
sns.set_style("whitegrid")
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 9}

plt.rc('font', **font)

plt.rcParams["font.serif"] = ["Times New Roman"]

pd.options.mode.chained_assignment = None  # Esto desactiva temporalmente la advertencia


######################LIMPIEZA DE DATOS ENTRENAMIENTO MODELO: Usamos dos datasets CTS y U012
#Importar los datos

### Uso esta funcion para importar los datos
def loader(path: str, lista: list):
    all_files = glob.glob(path + "\\" + "*.csv")
    all_files.sort()

    for filename in all_files:
        df = pd.read_csv(filename, sep=',')
        lista.append(df)

    return lista


# INTERQ
DLlist = []  # Datalogger
DLlist = loader(pathDL, DLlist)

#print(type(DLlist))

for i in range(len(DLlist)):  # Cleans double data
    DLlist[i] = DLlist[i][::2]



for DLdf in DLlist:
    for k in range(1, 10):
        DLdf.loc[k - 1, 'A.POS.Z'] = 0
        DLdf.loc[k - 1, 'V.A.POS.C'] = DLdf.loc[300, 'V.A.POS.C']
    DLdf['V.PLC.R[202]'] = (DLdf['V.PLC.R[202]'] + abs(DLdf['V.PLC.R[201]']))
    DLdf['V.PLC.R[205]'] = (abs(DLdf['V.PLC.R[205]']) + abs(DLdf['V.PLC.R[206]']))
    DLdf['V.PLC.R[211]'] = (abs(DLdf['V.PLC.R[211]']) + abs(DLdf['V.PLC.R[212]']))






# Torque signal of each stroke
def FigTorque(label, dfbroach, MediaEstable, y_max, y_min):
    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot()
    ax.grid(False)
    plt.plot(dfbroach['A.POS.Z'], dfbroach['V.PLC.R[202]'], label=label) #Torque signal of the stroke
    plt.axhline(y=MediaEstable, color='red', linestyle='--', label=f'Global Average: {MediaEstable:.2f}') #Average line of strokes
    plt.fill_between(
        dfbroach['A.POS.Z'],
        y_min,
        y_max,
        color='red',
        alpha=0.1,
        label=f'Torque Stable Range'
    ) # Stable range of previous passes
    plt.ylabel('Torque (Nm)')
    plt.xlabel('Pos. Z (mm)')
    plt.ylim([y_min*0.8, y_max*1.2])
    plt.legend(loc="lower left")
    plt.title('Torque signal')
    #plt.savefig(ruta + '/' +'TorqueCompleto_' + nombre+'.png') #To save the figure
    plt.show()




# Breakage
def AnalisisRotura(DLlist, pasada, y_min, y_max):


    pasada_rota = pasada  # The current and previous passes for comparison and detecting the broken tooth
    pasada_ref = pasada - 1
    pasadas_Analisis = [pasada_rota, pasada_ref]
    diferencia = []
    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot()
    ax.grid(False)
    for i,pasada in enumerate(pasadas_Analisis):
        dfbroach = DLlist[pasada].loc[
            (DLlist[pasada]['A.POS.Z'] >= zonaEstableInicio) & (DLlist[pasada]['A.POS.Z'] <= zonaEstableFin)
            ]
        label = f'{pasada + 1} broaching stroke'
        color = colores[i]
        plt.plot(dfbroach['A.POS.Z'], dfbroach['V.PLC.R[202]'], label=label, color=color)

    dfbroachRotura = DLlist[pasada_rota].loc[
        (DLlist[pasada_rota]['A.POS.Z'] >= zonaEstableInicio) & (DLlist[pasada_rota]['A.POS.Z'] <= zonaEstableFin)
        ]
    dfbroachRef = DLlist[pasada_ref].loc[
        (DLlist[pasada_ref]['A.POS.Z'] >= zonaEstableInicio) & (DLlist[pasada_ref]['A.POS.Z'] <= zonaEstableFin)
        ]

    label_PR = pasada_rota + 1
    print('Analizo en qué diente de la pasada ' + str(label_PR) + ' ha ocurrido la rotura')
    plt.title('Fracture Analysis stroke: ' + str((label_PR)))
    diferencia = abs(dfbroachRotura['V.PLC.R[202]'] - dfbroachRef['V.PLC.R[202]'])  # To detect the broken tooth, the maximum difference in torque signal for each tooth is estimated (Between current stroke and previous)
    maximos_diferencia = diferencia.nlargest(3)  # We retain the top 3 most likely ones
    indices_maximos = maximos_diferencia.index
    for j, indice_max_diferencia in enumerate(indices_maximos):  # To show the broken tooth number
        a_pos_z_max_diferencia = dfbroachRotura.loc[indice_max_diferencia, 'A.POS.Z']
        DienteRotura = int(round((float(a_pos_z_max_diferencia) - float(first_tooth_position)) / float(step)))
        print(DienteRotura)
        zRot1 = (DienteRotura - 1) * int(step) + int(first_tooth_position)
        zRot2 = (DienteRotura + 1) * int(step) + int(first_tooth_position)

        zRotText = (DienteRotura) * int(step) + int(first_tooth_position)

        plt.text(x=zRotText, y=(y_max + 3), s='Z' + str(DienteRotura), color='k')
        plt.axvline(x=zRot1, color='red', linestyle='--', alpha=0.5)
        plt.axvline(x=zRot2, color='red', linestyle='--', alpha=0.5)
        #plt.axvspan(zRot1, zRot2, alpha=0.05, color='red')


    indice_max_diferencia = diferencia.idxmax()
    a_pos_z_max_diferencia = dfbroachRotura.loc[indice_max_diferencia, 'A.POS.Z']
    DienteRotura = int(
        round(float(a_pos_z_max_diferencia) - float(first_tooth_position)) / float(step))  # Conversion of height to tooth number


    plt.ylabel('Torque (Nm)')
    plt.xlabel('Pos. Z (mm)')
    plt.ylim([y_min * 0.8, y_max * 1.2])
    plt.legend(loc="lower left")
    plt.show()


    return (DienteRotura)







rangestable = []
maxestable = []
minestable = []
mediaestable = []

# Crear un DataFrame para almacenar los resultados
summary_data = pd.DataFrame(columns=['Pasada', 'Max_Torque', 'Min_Torque', 'Mean_Torque', 'Amplitude'])

# Calcular y almacenar los resultados para cada pasada de la ZONA ESTABLE
for idx, DLdf in enumerate(DLlist):

    DLdf = DLdf.loc[(DLdf['A.POS.Z'] >= zonaEstableInicio) & (DLdf['A.POS.Z'] <= zonaEstableFin)]

    max_torque = DLdf['V.PLC.R[202]'].max()
    min_torque = DLdf['V.PLC.R[202]'].min()
    mean_torque = DLdf['V.PLC.R[202]'].mean()
    amplitude = max_torque - min_torque

    rangestable.append(amplitude)

    pasada = f'{idx + 1} broaching stroke'
    pas = idx
    print(pasada)
    media_rango = sum(rangestable) / len(rangestable)

    # si la amplitud es mayor al 27 % ha ocurrido una rotura
    if amplitude >= media_rango * MGnum:
        y_max = max(maxestable)
        y_min = min(minestable)
        print('Rotura en la pasada: ' + str (idx + 1))
        #FigTorque(pasada, DLdf, mean_torque, y_max, y_min)
        maxestable = []
        minestable = []
        mediaestable = []
        rangestable = []

        maxestable.append(max_torque)
        minestable.append(min_torque)
        mediaestable.append(mean_torque)
        rangestable.append(amplitude)

        dienteRotura = AnalisisRotura(DLlist, pas, y_min, y_max)
        print("Diente rotura: " + str(dienteRotura))

    else:
        maxestable.append(max_torque)
        minestable.append(min_torque)
        mediaestable.append(mean_torque)
        y_max = max(maxestable)
        y_min = min(minestable)
        #FigTorque(pasada, DLdf, mean_torque, y_max, y_min)



    print('-----------')




print("Transition: " + str(transition))