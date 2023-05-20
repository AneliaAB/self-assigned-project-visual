# %%
import tensorflow as tf
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import io
from PIL import Image
import os

# %%
# Set the path to the directory containing the colored and natural coral reef images
#data_dir = r'/kaggle/input/healthy-and-bleached-corals-image-classification'

#import zipfile
#with zipfile.ZipFile("/work/data/archive (1).zip") as zip_ref:
#    zip_ref.extractall("data")

# %%
data_dir = "../src/data/"
# %%
def is_image_corrupted(image_path):
    try:
        # Open the image file
        Image.open(image_path).verify()
        return False  # Image is not corrupted
    except (IOError, SyntaxError):
        return True  # Image is corrupted

def delete_corrupted_images(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if is_image_corrupted(file_path):
            print(f"Deleting corrupted image: {filename}")
            os.remove(file_path)
# %%
# Set the path to the directory containing the images
image_directory = '../src/data/bleached_corals/'
image_directory2 = '../src/data/healthy_corals/'

# %%
# Delete corrupted images
delete_corrupted_images(image_directory)
delete_corrupted_images(image_directory2)

# %%
# Define image dimensions, batch size, and train/test split ratio
img_width, img_height = 150, 150
batch_size = 32
validation_split = 0.2

# %%
# Create data generator with data augmentation
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=validation_split
)

# Load and split the data into train and test sets
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

test_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

# %%
# Create a CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# %%
from PIL import Image

# %%
# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
epochs = 10

# %%
H = model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=test_generator.samples // batch_size
)
# %%
import matplotlib.pyplot as plt
import numpy as np
def plot_history(H, epochs):
    plt.style.use("fivethirtyeight")
    plt.figure()
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss", linestyle=":")
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc", linestyle=":")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.tight_layout()
    plt.legend()
    plt.savefig("test.png", format="png") # specify filetype explicitly
    plt.show()

    plt.close()

plot_history(H, 10)

# %%
# Evaluate the model on the test set
_, accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {accuracy * 100:.2f}%")



# %%
