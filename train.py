import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import shutil

DATASET_DIR = "data"

def remove_thumbs(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower() == "thumbs.db":
                file_path = os.path.join(root, file)
                print(f"Lösche {file_path}")
                os.remove(file_path)

def validate_images(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                print(f"Ungültiger Dateityp entfernt: {file_path}")
                os.remove(file_path)
                continue
            try:
                with Image.open(file_path) as img:
                    img.verify()  # Überprüft die Integrität des Bildes
            except (IOError, SyntaxError, Image.DecompressionBombError, Image.UnidentifiedImageError) as e:
                print(f"Fehlerhaftes Bild gefunden: {file_path}")
                print(f"Fehler: {e}")
                return file_path  # Gibt den fehlerhaften Pfad zurück
            
def validate_and_collect_invalid_images(directory,error_dir="errors"):
    invalid_images = []  # Liste für ungültige Bilder
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                print(f"Ungültiger Dateityp verschoben: {file_path}")
                shutil.move(file_path, os.path.join(error_dir, file))
                continue
            try:
                with Image.open(file_path) as img:
                    img.verify()
            except Exception as e:
                print(f"Fehlerhaftes Bild gefunden: {file_path}")
                print(f"Fehler: {e}")
                invalid_images.append(file_path)
                shutil.move(file_path, os.path.join(error_dir, file))
                  # Fehlerhaften Pfad hinzufügen
    return invalid_images

# Daten prüfen und aufräumen
validate_images(os.path.join(DATASET_DIR, "cats"))
validate_images(os.path.join(DATASET_DIR, "dogs"))
remove_thumbs(DATASET_DIR)
invalid_image_paths = validate_and_collect_invalid_images("data")

# Fehlerhafte Bilder anzeigen
if invalid_image_paths:
    print("Folgende Bilder sind fehlerhaft:")
    for path in invalid_image_paths:
        print(path)
else:
    print("Keine fehlerhaften Bilder gefunden.")

# Bildgröße und Batch-Größe
IMG_SIZE = 128
BATCH_SIZE = 32

# Datenaugmentation und Normalisierung
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2,
)

train_data = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
)

valid_data = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
)

# Modell erstellen
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid"),
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Modell trainieren
try:
    history = model.fit(
    train_data,
    validation_data=valid_data,
    epochs=10,
)
except Exception as e:
    print(f"Trainingsfehler: {e}")

# Modell speichern
model.save("cats_vs_dogs_model.h5")
print("Modell gespeichert!")
