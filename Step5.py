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

testing_images = [
    "./Data/Test/Medium/Crack__20180419_06_19_09,915.bmp",
    "./Data/Test/Large/Crack__20180419_13_29_14,846.bmp"
]

class_labels = {0: "Small", 1: "Medium", 2: "Large", 3: "None"}
class_labelsperc = {0: "Small Crack", 1: "Medium Crack", 2: "Large Crack", 3: "No Crack"}

for img_path in testing_images:
    
    img, img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    actual_class = img_path.split("/")[-2]
    percentages = [f"{class_labelsperc[i]}: {predictions[0, i] * 100:.1f}%" for i in range(len(class_labelsperc))]
    plt.imshow(img)
    plt.title(f"True Crack Label: {actual_class}\nPredicted Crack Classification Label: {class_labels[predicted_class]}")

    for i, text in enumerate(percentages):
        plt.text(95, 95 - i * 5, text, fontsize=11, color='green', ha='right', va='top')

    plt.show()
