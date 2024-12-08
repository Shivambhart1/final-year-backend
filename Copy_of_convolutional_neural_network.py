#!/usr/bin/env python
# coding: utf-8

# # Convolutional Neural Network

# In[1]:


import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator


# In[2]:


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
training_set = train_datagen.flow_from_directory(
        'Labelled dataset/Train',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')


# In[3]:


test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
        'Labelled dataset/Test',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')


# In[4]:


cnn= tf.keras.models.Sequential()


# In[5]:


cnn.add(tf.keras.layers.Dense(units=9, activation='relu', input_shape=(64,64,3)))


# In[6]:


cnn.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu'))


# In[7]:


cnn.add(tf.keras.layers.MaxPool2D(pool_size= 2, strides=2,))


# In[8]:


cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu',))
cnn.add(tf.keras.layers.MaxPool2D(pool_size= 2, strides=2,))


# In[9]:


cnn.add(tf.keras.layers.Flatten())


# In[10]:


cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))


# In[11]:


cnn.add(tf.keras.layers.Dense(units=9, activation='softmax'))


# In[12]:


cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[13]:


cnn.fit(x= training_set, validation_data= test_set, epochs=5)


# In[ ]:


print(cnn.summary)


# Making a single prediction

# In[14]:


from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np

def prepare(imgage):
    IMG_SIZE= 64
    img_array= cv2.imread('ISIC_0034333.jpg',cv2.IMREAD_COLOR)
    new_array= cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)

y= cnn.predict([prepare('ISIC_0034333.jpg')])
print(y)
score = y[0]


# In[15]:


classes= list(training_set.class_indices)


# In[16]:


import numpy as np
n=np.argmax(y)


# In[20]:


disease=classes[n]
print(disease)


# In[18]:


cnn.save('Skin_disease.h5')


# In[19]:


x= plt.imread('ISIC_0034333.jpg')
plt.imshow(x)


# In[ ]:




