import tensorflow as tf
from tensorflow.keras import layers, models 
from tensorflow.keras.datasets import mnist
import numpy as np 
#import pandas as pd 
import matplotlib.pyplot as plt
#No es necesaria from PIL import Image

# Paso 2: Cargar el conjunto de datos MNIST (tiene 60 mil imágenes)
#Xtesr y y Test tuenen 10 mil imagenes de prueba.
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Paso 3: Preprocesar los datos
x_train = x_train.astype('float32') / 255  # Normalización
x_test = x_test.astype('float32') / 255
x_train = x_train.reshape(-1, 28 * 28)     # Aplanar las imágenes
x_test = x_test.reshape(-1, 28 * 28)
y_train = tf.keras.utils.to_categorical(y_train, 10)  # One-hot encoding
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Paso 4: Definir el modelo MLP.. construcción red neuronal
model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))  # 10 clases de salida

# Paso 5: Compilar el modelo
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# Paso 6: Entrenar el modelo
history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)
# Paso 7: Evaluar el modelo
test_loss, test_acc = model.evaluate(x_test, y_test)

#Paso 8: Guardar el modelo

model.save('mlp_model.h5') #Guardamos como archivo HDF5, lo que hace es que ya queda listo para poderse ejecutar desde cualquier interfaz
#Forma de guardar los datos de una manera más flexible, se visualiza bien para la librería que trabaja bien para la librería tensorflow
print("Modelo guardado")