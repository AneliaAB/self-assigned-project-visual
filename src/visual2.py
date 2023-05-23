# %%
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
#loading and loading images
import io
from PIL import Image
import os
#plotting
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#
from sklearn.metrics import classification_report

# Set the path to the directory containing the colored and natural coral reef images
# %%
data_dir = "../data/"

#deleting corrupted images 
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
bleached_directory = '../data/bleached_corals/'
healthy_directory = '../data/healthy_corals/'

# Delete corrupted images
delete_corrupted_images(bleached_directory)
delete_corrupted_images(healthy_directory)

#%%
#visualizing the data (https://problemsolvingwithpython.com/06-Plotting-with-Matplotlib/06.04-Saving-Plots/, https://www.kaggle.com/code/uysimty/keras-cnn-dog-or-cat-classification)
def create_df(path_to_folder):
    filenames = os.listdir(path_to_folder)

    categories = []
    for filename in filenames:
        if path_to_folder == "data/bleached_corals":
            categories.append(0)
        else:
            categories.append(1)

    df = pd.DataFrame({
        'filename': filenames,
        'category': categories
    })
    
    return df

healthy_df = create_df("data/healthy_corals")
bleached_df = create_df("data/bleached_corals")

df = pd.concat([bleached_df, healthy_df])
df_plt = df["category"].replace({0: 'bleached', 1: 'healthy'}).value_counts().plot.bar() 
df_plt.figure.savefig("../out/visualizing_dataframe.png", dpi=300, bbox_inches='tight') # specify filetype explicitly


# %%
# Define image dimensions, batch size, and train/test split ratio
img_width, img_height = 224, 224
batch_size = 32
validation_split = 0.2

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
# Load the pre-trained VGG16 model without the top (fully connected) layers
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Freeze the pre-trained layers
for layer in vgg_model.layers:
    layer.trainable = False

# Create a new model and add the pre-trained VGG16 model as a layer
model = Sequential()
model.add(vgg_model)

# Flatten the output of the VGG16 model
model.add(Flatten())

# Add a fully connected layer and an output layer
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
epochs = 10

H = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=test_generator.samples // batch_size
)

# %%
# Evaluate the model on the test set
_, accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# %%
predictions = model.predict(train_generator, batch_size=128)

# %%
#_________________________________________________________________________________
test_steps = len(test_generator)
y_pred = model.predict(test_generator, steps=test_steps)
y_pred = np.argmax(y_pred, axis=1)  # Convert probabilities to class labels

# Convert true labels from generator to array
test_generator.reset()
y_true = []
for i in range(test_steps):
    _, labels = test_generator.next()
    y_true.extend(labels)
y_true = np.array(y_true)

# Reshape or convert labels to 1-dimensional arrays
y_pred = y_pred.reshape(-1)
y_true = y_true.reshape(-1)

# Generate classification report
target_names = ['Natural', 'Colored']
report = classification_report(y_true, y_pred, target_names=target_names)

print(report)

report_path = "../out/classification_report.txt"

text_file = open(report_path, "w")
n = text_file.write(report)
text_file.close()

# %%
def plot_history(H, epochs):
    plt.style.use("seaborn-colorblind")

    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss", linestyle=":")
    plt.title("Loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc", linestyle=":")
    plt.title("Accuracy curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend()
    plt.savefig("../out/loss_accuracy_curve_corals.png", format="png") # specify filetype explicitly
    plt.show()

    plt.close()

plot_history(H, epochs)
# %%
