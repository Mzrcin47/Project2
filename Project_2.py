import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
#from scipy import misc

import os
from PIL import Image


' Step 1 '

desired_shape = (100, 100, 3)


#absolute_path= os.path.dirname(__file__)

train_data_dir='./Data/Train'
test_data_dir='./Data/Test'
validation_dir='./Data/Validation'


#train_data_dir= os.path.join(absolute_path, 'Data', 'Train')
#test_data_dir=os.path.join(absolute_path, 'Data', 'Test')
#validation_dir=os.path.join(absolute_path, 'Data', 'Validation')
#extracted_and_preprocessed_images = extract_and_preprocess_images_from_folder(data_dir)

train_images = os.listdir(train_data_dir)
validation_images = os.listdir(validation_dir)
test_images = os.listdir(test_data_dir)

'data augmantation and imagedatafounddirectory'

train_datagen = ImageDataGenerator(
    shear_range=0.25,     
    zoom_range=0.25       
)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,         
    target_size= desired_shape[:2], 
    batch_size=32,          
    class_mode='categorical' 
    )

validation_datagen = ImageDataGenerator(rescale=1.0/255.0)

validation_gen = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=desired_shape[:2],
    batch_size=32,
    class_mode='categorical'
)

calidation_gen = train_datagen.flow_from_directory(
    train_data_dir,         
    target_size= desired_shape[:2], 
    batch_size=32,          
    class_mode='categorical' 
    )

'Step 2/3'

model1 = models.Sequential()
model1.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model1.add(layers.MaxPooling2D((2, 2)))

model1.add(layers.Conv2D(64, (3, 3), activation='relu'))
model1.add(layers.MaxPooling2D((2, 2)))
model1.add(layers.Conv2D(64, (3, 3), activation='relu'))
#model1.summary()

model1.add(layers.Flatten())
model1.add(layers.Dense(64, activation='relu'))
model1.add(layers.Dense(4, activation='softmax'))

model1.summary()


model1.compile(optimizer='adam',
               loss='categorical_crossentropy',
               metrics=['accuracy'])

history = model1.fit(train_generator, epochs=10, validation_data=validation_gen)


#print(history.history)


#plt.plot(history.history['accuracy'], label='accuracy')
#plt.plot(history.history['val_accuracy'], label='val_accuracy')
