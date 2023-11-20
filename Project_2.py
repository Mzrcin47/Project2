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
    rescale=1./255,
    shear_range=0.2,     
    zoom_range=0.2
)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,        
    target_size= (100,100), 
    batch_size=32,          
    class_mode='categorical' 
    )

validation_datagen = ImageDataGenerator(rescale=1.0/255.0)

validation_gen = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(100,100),
    batch_size=32,
    class_mode='categorical'
)

'Step 2/3'

#model1 = models.Sequential()
#model1.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
#model1.add(layers.MaxPooling2D((2, 2)))

#model1.add(layers.Conv2D(64, (3, 3), activation='relu'))
#model1.add(layers.MaxPooling2D((2, 2)))
#model1.add(layers.Conv2D(64, (3, 3), activation='relu'))
#model1.summary()

##model1.add(layers.Flatten())
#model1.add(layers.Dense(64, activation='relu'))
#model1.add(layers.Dense(4, activation='softmax'))

#model1.summary()


#odel1.compile(optimizer='adam',
  #             loss='categorical_crossentropy',
   #            metrics=['accuracy'])

#history = model1.fit(train_generator, epochs=10, validation_data=validation_gen)


# print(history.history)

model2 = models.Sequential()
model2.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(desired_shape)))
model2.add(layers.MaxPooling2D((2, 2)))
model2.add(layers.Conv2D(64, (3, 3), activation='relu'))
model2.add(layers.MaxPooling2D((2, 2)))
model2.add(layers.Conv2D(128, (3, 3), activation='relu'))
model2.add(layers.Dropout(0.5))


model2.add(layers.Flatten())
model2.add(layers.Dense(128, activation='relu'))
model2.add(layers.Dense(4, activation='softmax'))


model2.summary()


model2.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model2.fit(train_generator, epochs=10, validation_data=validation_gen)

model2.save('model2.h5')

print(history.history)

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

# Plot training loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.show()

test_loss, test_acc = model2.evaluate(train_generator)
val_loss, val_acc = model2.evaluate(validation_gen)
print("Test Accuracy:", test_acc)
print("Validation Accuracy:", val_acc)



