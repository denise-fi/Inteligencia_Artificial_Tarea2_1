import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Cargar los datos desde el archivo CSV
data = pd.read_csv('datos.csv')

# Separar las variables independiente (altura) y dependiente (peso)
X = data['altura'].values.reshape(-1, 1)  # Altura como matriz de una columna
y = data['peso'].values  # Peso

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo de regresión lineal
modelo = LinearRegression()

# Entrenar el modelo con los datos de entrenamiento
modelo.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = modelo.predict(X_test)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Puntuación modelo:", r2)
print("Error:", mse)

# Ejemplo de predicción con una altura específica
n_altura = np.array([[1.75]])
peso_predicho = modelo.predict(n_altura)
print(f"El peso predicho para una altura  es de {n_altura[0][0]} m es: {peso_predicho[0]:.2f} kg")

#Respuesta 
#¿Funciona bien o no?
#En base al peso ideal si funciona bien, pero no para una persona que tiene desnutricion o sobrepeso.
