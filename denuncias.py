
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