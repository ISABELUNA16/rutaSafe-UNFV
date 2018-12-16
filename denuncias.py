
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

data2 = pd.read_csv("./denuncias_violencia_familiar2.csv", encoding = "ISO-8859-1")

departamentos = []
años = []
denuncias = []

for index, row in data2.iterrows():
    i = 0
    for col in row:
        if i > 0 and index > 0:
            departamentos.append(index)
            años.append(i)
            denuncias.append(col)

        if i > 12 and i < 16 and index > 12 and index < 16:
            print(col)

        i = i + 1

dataX2 = pd.DataFrame()
dataX2["departamentos"] = departamentos
dataX2["años"] = años

#training
XY_train = np.array(dataX2)
z_train = np.array(denuncias)
print(z_train)

regresion = linear_model.LinearRegression()

regresion.fit(XY_train, z_train)

z_pred = regresion.predict(XY_train)

# Veamos los coeficienetes obtenidos, En nuestro caso, serán la Tangente
print('Coefficients: \n', regresion.coef_)
# Este es el valor donde corta el eje Y (en X=0)
print('Independent term: \n', regresion.intercept_)
# Error Cuadrado Medio
print("Mean squared error: %.2f" % mean_squared_error(z_train, z_pred))
# Puntaje de Varianza. El mejor puntaje es un 1.0
print('Variance score: %.2f' % r2_score(z_train, z_pred))

print('z_train')
print(z_train)
print('z_pred')
print(z_pred)

fig = plt.figure()
ax = Axes3D(fig)

# Creamos una malla, sobre la cual graficaremos el plano
xx, yy = np.meshgrid(np.linspace(0, 3500, num=10), np.linspace(0, 60, num=10))

# calculamos los valores del plano para los puntos x e y
nuevoX = (regresion.coef_[0] * xx)
nuevoY = (regresion.coef_[1] * yy)

# calculamos los correspondientes valores para z. Debemos sumar el punto de intercepción
z = (nuevoX + nuevoY + regresion.intercept_)

# Graficamos el plano
#.plot_surface(xx, yy, z, alpha=0.2, cmap='hot')

# Graficamos en azul los puntos en 3D
ax.scatter(XY_train[:, 0], XY_train[:, 1], z_train, c='blue', s=30)

# Graficamos en rojo, los puntos que
ax.scatter(XY_train[:, 0], XY_train[:, 1], z_pred, c='red', s=40)

# con esto situamos la "camara" con la que visualizamos
ax.view_init(elev=30., azim=65)

ax.set_xlabel('Departamento')
ax.set_ylabel('Año')
ax.set_zlabel('Cantidad de Denuncias por violencía familiar')
ax.set_title('Regresión Lineal para el modelo de denuncias')

plt.show()