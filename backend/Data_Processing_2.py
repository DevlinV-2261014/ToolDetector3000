import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.layers import BatchNormalization
from keras.regularizers import l2
import numpy as np
import os
import shutil

from keras.src.callbacks import ModelCheckpoint
from keras.src.optimizers import SGD
from sklearn.model_selection import train_test_split


class_names = ['CombWrench', 'Hammer', 'Screwdriver', 'Wrench']
# Functions
def augment(image, label):
    # Random flips
    # image = tf.image.random_flip_left_right(image)
    # image = tf.image.random_flip_up_down(image)

    # Random rotation
    # image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))

    # Random brightness and contrast
    #image = tf.image.random_brightness(image, max_delta=0.2)
    #image = tf.image.random_contrast(image, lower=0.8, upper=1.2)

    # Random saturation and hue
    # image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    # image = tf.image.random_hue(image, max_delta=0.2)

    # Ensure the pixel values are still in [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label

def preprocess_image(image, label):
    image = tf.image.resize(image, image_shape)        # Resize the image to 224x224
    image = tf.image.random_flip_left_right(image)    # Random horizontal flip
    image = tf.image.random_brightness(image, max_delta=0.1)  # Adjust brightness randomly
    image /= 255.0                                    # Normalize pixel values
    return image, label

# Get class distribution
def count_samples_per_class(ds):
    # Initialize the count dictionary
    count_dict = {class_name: 0 for class_name in class_names}

    # Iterate over the dataset
    for images, labels in ds.unbatch().as_numpy_iterator():
        # Update the count dictionary
        count_dict[class_names[np.argmax(labels)]] += 1

    return count_dict


def configure_for_performance(ds):
    ds = ds.cache()
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds

def data_process(base_dir, train_dir, val_dir):
    # Clean up previous data folders if they exist
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    if os.path.exists(val_dir):
        shutil.rmtree(val_dir)

    # Create fresh train and validation directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # List of classes
    classes = ['CombWrench', 'Hammer', 'Screwdriver', 'Wrench']

    # Process each class
    total_size = 8000
    validation_size = int(total_size * 0.1 / 4)  # 50% of the data for validation
    for cls in classes:
        # Create directories for each class within train and validation folders
        class_train_dir = os.path.join(train_dir, cls)
        class_val_dir = os.path.join(val_dir, cls)
        os.makedirs(class_train_dir, exist_ok=True)
        os.makedirs(class_val_dir, exist_ok=True)

        # Source directory for the class
        class_dir = os.path.join(base_dir, cls)

        # Get a list of all files in the class directory
        files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]

        # Split the files into training and validation sets
        train_files, val_files = train_test_split(files, test_size=validation_size)  # Ensures 5 in validation

        # Move files to their new directories
        for f in train_files:
            shutil.copy(f, class_train_dir)
        for f in val_files:
            shutil.copy(f, class_val_dir)
    # Create a dataset
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        image_size=image_size,
        batch_size=40,
        label_mode='categorical'
    )

    validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        val_dir,
        image_size=image_size,
        batch_size=40,
        label_mode='categorical'
    )

    return train_dataset, validation_dataset

def data_preprocess(train_dataset, validation_dataset, augment=False):
    # Apply the preprocessing and augmentation functions
    if augment:
        train_dataset = train_dataset.map(lambda x, y: preprocess_image(x, y)).map(augment,
                                                                                   num_parallel_calls=tf.data.AUTOTUNE)
        validation_dataset = validation_dataset.map(lambda x, y: preprocess_image(x, y))  # No augmentation for validation
    else:
        train_dataset = train_dataset.map(lambda x, y: preprocess_image(x, y))
        validation_dataset = validation_dataset.map(lambda x, y: preprocess_image(x, y))

    # Calculate the class distribution
    train_class_distribution = count_samples_per_class(train_dataset)
    validation_class_distribution = count_samples_per_class(validation_dataset)

    print('Training set class distribution:', train_class_distribution)
    print('Validation set class distribution:', validation_class_distribution)

    # Configure the datasets for performance
    train_dataset = configure_for_performance(train_dataset)
    validation_dataset = configure_for_performance(validation_dataset)

    return train_dataset, validation_dataset


# Base directory where your classes are stored
base_dir = 'Data/All'
# Prepare the directory structure for training and validation sets
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'validation')

image_shape = [212,212]
image_size = (212, 212)

# train_dataset, validation_dataset = data_process(base_dir, train_dir, val_dir)

# Right after this, save the class names
# class_names = train_dataset.class_names


# Apply the preprocessing function
# train_dataset = train_dataset.map(preprocess_image)
# validation_dataset = validation_dataset.map(preprocess_image)

# train_dataset, validation_dataset = data_preprocess(train_dataset, validation_dataset)

# Adjusted learning rate schedule
initial_learning_rate = 0.01
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=432,  # Adjusted to decay approximately every 4 epochs
    decay_rate=0.85,
    staircase=True
)
# optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
optimizer = keras.optimizers.Adam(learning_rate=0.01)
# optimizer = SGD(learning_rate=lr_schedule, momentum=0.9)

# Define the model
# Define the CNN model
model = Sequential([
    Conv2D(8, (3, 3), activation='relu', padding='same', input_shape=(212, 212, 3)),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Conv2D(4, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(4, activation='softmax')
])

model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# Early stopping and model checkpointing
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3, min_lr=0.0001)
model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, verbose=1)
# Create training loops
loops = 1
new_training = True
for i in range(loops):
    print("Loop: ", i + 1)
    train_dataset, validation_dataset = data_process(base_dir, train_dir, val_dir)
    train_dataset, validation_dataset = data_preprocess(train_dataset, validation_dataset)
    # Load the best model if it exists
    if not new_training:
        try:
            model.load_weights('best_model.keras')
            print("Loaded best model from previous iteration.")
        except:
            print("Starting training without pre-loaded model weights.")
    history = model.fit(train_dataset, validation_data=validation_dataset, epochs=20, callbacks=[early_stopping, reduce_lr, model_checkpoint])
    new_training = False

# Save the trained model
model.save('tool_classifier_model.keras')