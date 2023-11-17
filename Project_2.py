import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential

import os
from PIL import Image


' Step 1 '

#desired_shape = (100, 100, 3)
width,height,channels= 100,100,3 

#absolute_path= os.path.dirname(__file__)

train_data_dir='./Data/Train'
test_data_dir='./Data/Test'
validation_dir='./Data/Validation'

#traindataset= tf.keras.utils.image_dataset_from_directory('
 #   validation_dir
  #  )

#plt.figure(figsize=(10,10))
#for images, labels in train_data_dir.take(1):
 #   plt.imshow(images[1].numpy().asytype("unit8"))
 #   plt.axis("off")

#train_data_dir= os.path.join(absolute_path, 'Data', 'Train')
#test_data_dir=os.path.join(absolute_path, 'Data', 'Test')
#validation_dir=os.path.join(absolute_path, 'Data', 'Validation')
#extracted_and_preprocessed_images = extract_and_preprocess_images_from_folder(data_dir)

#train_images = os.listdir(train_data_dir)
#validation_images = os.listdir(validation_dir)
#test_images = os.listdir(test_data_dir)

'data augmantation and imagedatafounddirectory'

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0, 
    shear_range=0.25,     
    zoom_range=0.25       
)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,         
    target_size=(100, 100, 3), 
    batch_size=32,          
    class_mode='categorical' 
    )

validation_datagen = ImageDataGenerator(rescale=1.0/255.0)

calidation_gen = train_datagen.flow_from_directory(
    train_data_dir,         
    target_size=(100, 100, 3), 
    batch_size=32,          
    class_mode='categorical' 
    )

'Step 2'

model1 = models.Sequential()
model1.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model1.add(layers.MaxPooling2D((2, 2)))
model1.add(layers.Conv2D(64, (3, 3), activation='relu'))
model1.add(layers.MaxPooling2D((2, 2)))
model1.add(layers.Conv2D(64, (3, 3), activation='relu'))
#model1.summary()

model1.add(layers.Flatten())
model1.add(layers.Dense(64, activation='relu'))
model1.add(layers.Dense(10))

model1.summary()

#model1.compile(optimizer='adam',
 #             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  #            metrics=['accuracy'])

#history = model1.fit(train_images, epochs=10, validation_data=validation_datagen)

