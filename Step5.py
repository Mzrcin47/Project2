import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

model_path = 'model2.h5'
model2 = load_model(model_path)


test_data_dir_Medium='./Data/Test/Small'
test_data_dir_Large='./Data/Test/Large'


test_images_Medium = os.listdir(test_data_dir_Medium)
test_images_Large = os.listdir(test_data_dir_Large)


test_images = ['Crack__20180419_06_19_09,915.bmp', 'Crack__20180419_13_29_14,846.bmp']
