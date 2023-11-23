import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


model = load_model('model2.h5')

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(100, 100))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img, img_array

testing_images = ["./Data/Test/Medium/Crack__20180419_06_19_09,915.bmp",
               "./Data/Test/Large/Crack__20180419_13_29_14,846.bmp"]

