#!/usr/bin/env python
# coding: utf-8

# In[85]:


# Author Muhammad Abdullah
# Importing the Required Python Libraries

import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import split_folder as sf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[8]:


# Loading the Dataset
# Input_folder is the Dataset path

input_folder = "/Users/muhammadabdullah/Desktop/Studies/MS AI UEL/AI & Computer Vision/Project/archive/natural_images"

# Output is the to save the Train, Valid and Test

output = "/Users/muhammadabdullah/Desktop/Studies/MS AI UEL/AI & Computer Vision/Project/archive/Input_Data" 

# Splitting the dataset into Testing Validation and Testing with the ratio of 70%, 15% and 15%

sf.ratio(input_folder, output=output, seed=42, ratio=(.7, .15, .15)) 


# In[113]:


IMAGE_SIZE = [150, 150]

train_path = '/Users/muhammadabdullah/Desktop/Studies/MS AI UEL/AI & Computer Vision/Project/archive/Input_Data/train'
valid_path = '/Users/muhammadabdullah/Desktop/Studies/MS AI UEL/AI & Computer Vision/Project/archive/Input_Data/val'
test_path = '/Users/muhammadabdullah/Desktop/Studies/MS AI UEL/AI & Computer Vision/Project/archive/Input_Data/test'


from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(train_path,
                                                 target_size = (150, 150),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory(valid_path,
                                            target_size = (150, 150),
                                            batch_size = 32,
                                            class_mode = 'categorical')

test_data = test_datagen.flow_from_directory(test_path,
                                            target_size = (150, 150),
                                            batch_size = 32,
                                            class_mode = 'categorical')


# In[117]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
#defining model
model=models.Sequential()
#adding convolution layer
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
#adding pooling layer
model.add(MaxPool2D(2,2))
#adding convolution layer
model.add(Conv2D(64,(3,3),activation='relu'))
#adding pooling layer
model.add(MaxPool2D(2,2))
#adding convolution layer
model.add(Conv2D(128,(3,3),activation='relu'))
#adding pooling layer
model.add(MaxPool2D(2,2))
#adding convolution layer
model.add(Conv2D(128,(3,3),activation='relu'))
#adding pooling layer
model.add(MaxPool2D(2,2))
#adding fully connected layer
model.add(Flatten())
model.add(Dense(512,activation='relu'))
#adding output layer
model.add(Dense(8,activation='sigmoid'))
#compiling the model
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
# fit the model
r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=50,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)


# In[123]:


# loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

plt.savefig('LossVal_loss')

# accuracies
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()
plt.savefig('AccVal_acc')


# In[121]:


test_loss, test_acc = model.evaluate(test_data)


# In[ ]:




