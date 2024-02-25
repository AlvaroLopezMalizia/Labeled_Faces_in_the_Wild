#!/usr/bin/env python
# coding: utf-8

# # TP Final de Aprendizaje Profundo

# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_lfw_people.html

# In[1]:


import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people

# Utilizaremos solo imagenes de 7 personas con mas de 70 imagenes disponibles.
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.5, color=False, download_if_missing=True)

# Inspeccion de los tamaños
n_samples, h, w = lfw_people.images.shape

# Datos sin divir en subconjuntos
X = lfw_people.images

# Etiquetas y clases
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("Total dataset size:")
print("h, w: (%d, %d)" % (h, w))
print("n_samples: %d" % n_samples)
print("n_classes: %d" % n_classes)


# In[2]:


# Algunas funciones para graficar
def plot_gallery(images, number, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=0.01, right=0.99, top=0.90, hspace=0.35)
    for i in range(number):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i], cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())

def title(y, target_names):
    return "%s" % (target_names[y])


# In[3]:


titles = [title(y[i], target_names) for i in range(20)]
plot_gallery(X, 12 , titles, h, w)


# In[4]:


#Features: numeros reales, entre 0 y 255
print(X[0])


# # RESOLUCIÓN TP FINAL DEL CURSO 
# 
# ALVARO LOPEZ MALIZIA

# ## Cargo las librerías

# In[5]:


import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt

import tensorflow as tf


# In[6]:


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

from tensorflow.keras.metrics import MSE
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.losses import categorical_crossentropy

from tensorflow.keras.utils import to_categorical


# In[23]:


from sklearn.model_selection import KFold


# ## Separación de los datos en el conjunto de train y test

# In[7]:


#Pongo semilla para separar siempre en train y test de la misma forma
np.random.seed(14)

# Cargo los datos
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.5, color=False, download_if_missing=True)


# Datos sin divir en subconjuntos
X = lfw_people.images

# Etiquetas y clases
y = lfw_people.target


#Separo el conjunto de datos en train y test con una proporción 80:20
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42,
                                                    shuffle=True) #Con shuffle me garantizo que esté estratificado


# ## Preparacion de los datos

# In[8]:


#Obtengo el total de las clases
n_clases=np.max(np.unique(lfw_people.target))+1
print('El total de las clases es',n_clases)


# In[9]:


#Inspeciono los valores
x_train[0]


# In[10]:


#YA ESTÁN COMPRENDIDOAS ENTRE 0 Y 1
#Por lo que no ejecuto esta parte del codigo
# Paso los datos de los bits de las imágenes, para que los valores estén comprendidos entre [0,1]
#x_train=x_train.astype('float32')/255.0 

#x_test=x_test.astype('float32')/255.0  


# In[11]:


#Inspeciono los elementos
y_train[0]


# In[12]:


# Paso a one-hot coding las etiquetas de ambos conjuntos

y_train=to_categorical(y_train,n_clases)

y_test=to_categorical(y_test,n_clases)


# In[13]:


y_train[0] #Chequeo que esté correcto el one-hot


# ## Modelo

# In[21]:


#Voy a trabajar con un modelo de redes convolucionales
#A diferencia de las redes densas, no tengo que hacer el paso de flatten. Voy directo con las imagenes 2D

#Parametros de entrenamiento (Sin optimizar)
lr = 1.0
epochs = 30 #
batch_size = 32
k = 5


# In[15]:


x_train.shape #Chequeo las dimensiones son 62 pixeles de alto por 47 de ancho,
              #y el valor 1030 corresponder a numero de muestras del entrenamiento


# In[16]:


# Model (Uso la misma arquitectura que vimos en clase)
#---------------------------------------------------------------------#
input_layer = Input(shape=(62,47,1))# Como es un solo canal le pongo 1 
conv_1 = Conv2D(32, (3, 3), activation='relu') (input_layer)
conv_2 = Conv2D(64, (3, 3), activation='relu') (conv_1)
pool_1 = MaxPooling2D(pool_size=(2, 2)) (conv_2)
dropout_1 = Dropout(0.25) (pool_1)
flatten_1 = Flatten() (dropout_1)
dense_1 = Dense(100, activation='relu') (flatten_1)
dropout_2 = Dropout(0.25) (dense_1)
output_layer = Dense(n_classes, activation='softmax') (dropout_2)
#---------------------------------------------------------------------#
model_conv = Model(input_layer, output_layer)


# In[17]:


#Defino el optimizador y las métricas
Adadelta_optimizer = Adadelta(learning_rate=lr, rho=0.95)
model_conv.compile(optimizer=Adadelta_optimizer, loss='categorical_crossentropy', metrics=['acc', 'mse'])
model_conv.summary()


# ## Entrenamiento del modelo

# In[18]:


# VOY VIENDO EL TIEMPO DE ENTRENAMIENTO DEL MODELO
start_time = time.time()
history_conv = model_conv.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test), shuffle=True, verbose=1)
end_time = time.time()
print('\nElapsed Dense Model training time: {:.5f} seconds'.format(end_time-start_time))


# ## Evaluo el proceso de entrenamiento

# In[19]:


f = plt.figure(figsize=(10,8))

plt.subplot(1,2,1)
plt.plot(history_conv.history['acc'], linewidth=3, label='Train Accuracy')
plt.plot(history_conv.history['val_acc'], linewidth=3, label='Val Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')


plt.subplot(1,2,2)
plt.plot(history_conv.history['mse'], linewidth=3, label='Train Accuracy')
plt.plot(history_conv.history['val_mse'], linewidth=3, label='Val Accuracy')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend(loc='lower right')

plt.show()


# In[ ]:


# Se observa que ya a partir de las 15 epocas, hay un overffiting del modelo


# # PRUEBA FINAL CON FOTO MÍA

# In[25]:


# Cargo una foto mia


# In[27]:


# importing libraries.
from PIL import Image
 
# storing image path
fname = 'Foto.png'
 
    
# La paso a blanco y negro
image = Image.open(fname).convert("L")
 
# mapping image to gray scale
plt.imshow(image, cmap='gray')
plt.show()


# In[32]:


np.shape(image) #Veo cuantos pixeles tiene


# In[34]:


# le Bajo los pixeles
# Redimensionar la imagen a 62 x 47 píxeles
new_size = (47,62)
resized_image = image.resize(new_size)

# Mostrar la imagen redimensionada
plt.imshow(resized_image, cmap='gray')
plt.show()


# In[39]:


#Paso la imagen a array de numpy con valores entre 0-1

imagen_mia=np.array(resized_image)

imagen_mia=imagen_mia.astype('float32')/255.0


# In[45]:


imagen_mia.shape


# In[49]:


imagen_mia=imagen_mia.reshape((1, 62, 47, 1))


# In[51]:


# Le aplico el modelo
probabilidad = model_conv.predict(imagen_mia,verbose=1)
probabilidad


# In[60]:


posicion_maximo = np.argmax(probabilidad)
#Segun el modelo
print('Segun el modelo, me parezco más a la clase N°:',posicion_maximo)


# In[58]:


# Filtrar las imágenes y etiquetas para la clase 1
clase_1_indices = np.where(y == 1)[0]
imagenes_clase_1 = X[clase_1_indices]
etiquetas_clase_1 = y[clase_1_indices]

# Mostrar algunas imágenes de la clase 1
n_imagenes_a_mostrar = 5
fig, axes = plt.subplots(1, n_imagenes_a_mostrar, figsize=(15, 3))

for i in range(n_imagenes_a_mostrar):
    axes[i].imshow(imagenes_clase_1[i], cmap='gray')
    axes[i].set_title(f'Clase 1 - Persona {etiquetas_clase_1[i]}')
    axes[i].axis('off')

plt.show()


# In[ ]:




