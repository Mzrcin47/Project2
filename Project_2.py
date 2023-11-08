import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import os
from PIL import Image


' Step 1 '

desired_shape = (100, 100, 3)

absolute_path= os.path.dirname(__file__)

train_data_dir= os.path.join(absolute_path, 'Data', 'Train')
test_data_dir=os.path.join(absolute_path, 'Data', 'Test')
validation_dir=os.path.join(absolute_path, 'Data', 'Validation')
#extracted_and_preprocessed_images = extract_and_preprocess_images_from_folder(data_dir)

train_images = os.listdir(train_data_dir)
validation_images = os.listdir(validation_dir)
test_images = os.listdir(test_data_dir)

