import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

model_path = 'model2.h5'
model2 = load_model(model_path)

train_data_dir='./Data/Train'
test_data_dir='./Data/Test'
validation_dir='./Data/Validation'

train_images = os.listdir(train_data_dir)
validation_images = os.listdir(validation_dir)
test_images = os.listdir(test_data_dir)