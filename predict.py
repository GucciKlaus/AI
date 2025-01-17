import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Modell laden
model = tf.keras.models.load_model("cats_vs_dogs_model.h5")

def predict_image(image_path):
    # Bild laden und vorbereiten
    img = load_img(image_path, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0  # Normalisierung
    img_array = np.expand_dims(img_array, axis=0)  # Batch-Dimension hinzufügen

    # Vorhersage
    prediction = model.predict(img_array)
    if prediction[0][0] > 0.5:
        return "Hund"
    else:
        return "Katze"

# Test mit einem neuen Bild
image_path = "data/Dog/12459.jpg"  # Pfad zu einem Bild
result = predict_image(image_path)
print(f"Das Bild zeigt: {result}")
