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

class_labels = {0: "Small", 1: "Medium", 2: "Large", 3: "None"}

for img_path in testing_images:
    img, img_array = preprocess_image(img_path)

    predictions = model.predict(img_array)

    predicted_class = np.argmax(predictions)
    confidence = predictions[0, predicted_class]

    actual_class = img_path.split("/")[-2]

    plt.imshow(img)
    plt.title(f"True Crack Classification Label: {actual_class}\nPredicted Crack Classification Label: {class_labels[predicted_class]}")
    plt.show()