
# Imports necesarios
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
#plt.rcParams['figure.figsize'] = (16, 9)
#plt.style.use('ggplot')
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("./denuncias_por_delito.csv", encoding = "ISO-8859-1")
data_habitantes = pd.read_csv("./habitantes_por_departamento.csv", encoding = "ISO-8859-1")

departamentos = []
años = []
denuncias = []
#habitantess_por_departamentos = [379384, 1083519, 405759, 1382730, 616176, 1341012, 994494, 1205527, 347639, 721047, 850765, 1246038, 1778080, 1197260, 9485405, 883510, 141070, 174863, 254065, 1856809, 1172697, 813381, 329332, 224863, 496459]
habitantess_por_departamentos = []

#Leyendo numero de habitantes por departamentos
for index, row in data_habitantes.iterrows():
    if index > 0:
        habitantess_por_departamentos.append(row[1])

#Leyendo data de delitos
index = 0
for index, row in data.iterrows():
    i = 0
    for col in row:
        if i > 0 and index > 0:
            departamentos.append(habitantess_por_departamentos[index-1])
            años.append(2004 + i)
            denuncias.append(col)
        i = i + 1

dataX2 = pd.DataFrame()
dataX2["departamentos"] = departamentos
dataX2["años"] = años

#training
XY_train = np.array(dataX2)
z_train = np.array(denuncias)

regresion = linear_model.Ridge(alpha=.5)

regresion.fit(XY_train, z_train)

z_pred = regresion.predict(XY_train)

#print(regresion.score(XY_train, z_pred))

# Veamos los coeficienetes obtenidos, En nuestro caso, serán la Tangente
print('Coefficients: \n', regresion.coef_)
# Este es el valor donde corta el eje Y (en X=0)
print('Independent term: \n', regresion.intercept_)
# Error Cuadrado Medio
print("Mean squared error: %.2f" % mean_squared_error(z_train, z_pred))
# Puntaje de Varianza. El mejor puntaje es un 1.0
print('Variance score: %.2f' % r2_score(z_train, z_pred))


departamentos_new = []
años_new = []

for index, row in data.iterrows():
    i = 0
    for i in range(2018, 2028):
        departamentos_new.append(habitantess_por_departamentos[index-1])
        años_new.append(i)

dataX2_new = pd.DataFrame()
dataX2_new["departamentos"] = departamentos_new
dataX2_new["años"] = años_new

#training
XY_new = np.array(dataX2_new)

z_new = regresion.predict(XY_new)

fig = plt.figure()
ax = Axes3D(fig)

# Creamos una malla, sobre la cual graficaremos el plano
xx, yy = np.meshgrid(np.linspace(0, 10000, num=10), np.linspace(2004, 2030, num=10))

# calculamos los valores del plano para los puntos x e y
nuevoX = (regresion.coef_[0] * xx)
nuevoY = (regresion.coef_[1] * yy)

# calculamos los correspondientes valores para z. Debemos sumar el punto de intercepción
z = (nuevoX + nuevoY + regresion.intercept_)

# Graficamos el plano
ax.plot_surface(xx, yy, z, alpha=0.2, cmap='hot')

# Graficamos en azul los puntos en 3D
ax.scatter(XY_train[:, 0], XY_train[:, 1], z_train, c='blue', s=2, label='Entrenamiento: año 2005 - 2017')

# Graficamos en rojo, los puntos que
ax.scatter(XY_train[:, 0], XY_train[:, 1], z_pred, c='red', s=2, label='Predicción del entrenamiento: año 2005 - 2017')

# Graficamos en rojo, los puntos que
ax.scatter(XY_new[:, 0], XY_new[:, 1], z_new, c='orange', s=2, label='Predicción futura: año 2018 - 2028')

ax.legend()

# con esto situamos la "camara" con la que visualizamos

#Vista inclinada
ax.view_init(elev=50., azim=45)

#Vista frontal con la perspectiva en los años
ax.view_init(elev=0., azim=5)

ax.set_xlabel('Departamentos (Número de habitantes)')
ax.set_ylabel('Año')
ax.set_zlabel('Denuncias por Comisión de delitos')
ax.set_title('Regresión Lineal', loc='left')

ax.set

plt.show()