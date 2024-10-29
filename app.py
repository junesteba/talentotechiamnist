import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Cargar el modelo previamente entrenado 
model = tf.keras.models.load_model(r'C:\Users\Usuario\Desktop\IA\Talento Tech\Entorno virtual ANACONDA\mlp_model.h5')

# Función para preprocesar la imagen
def preprocess_image(image):
    # Convertir la imagen a escala de grises y redimensionar a 28x28
    image = image.convert('L')  # Convertir la imagen a escala de grises
    image = image.resize((28, 28))  # Redimensionar a 28x28
    image = np.array(image)  # Convertir a array de numpy
    image = image / 255.0  # Normalizar a valores entre 0 y 1
    image = np.reshape(image, (1, 28*28))  # Redimensionar para la entrada del modelo
    return image

# Título de la aplicación
st.title('Clasificación de dígitos manuscritos')

# Cargar la imagen
uploaded_file = st.file_uploader("Cargar una imagen de un dígito (0-9)", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # Mostrar la imagen cargada
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen cargada", use_column_width=True)
    
    # Preprocesar la imagen 
    processed_image = preprocess_image(image)
    
    # Hacer la predicción
    prediction = model.predict(processed_image)
    predicted_digit = np.argmax(prediction)
    
    # Mostrar la predicción
    st.write(f"Predicción: **{predicted_digit}**")
    
    # Mostrar las probabilidades
    for i in range(10):
        st.write(f"Dígito {i}: {prediction[0][i]:.4f}")
    
    # Mostrar imagen procesada para ver el preprocesamiento
    plt.imshow(processed_image.reshape(28, 28), cmap='gray')  # Convierte a 28x28 para visualizar
    plt.axis('off')
    st.pyplot(plt)
     