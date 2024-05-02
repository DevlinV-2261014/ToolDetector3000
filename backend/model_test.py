import keras
import numpy as np
import tensorflow as tf
from keras.models import load_model
import os
from PIL import Image

x = 64
y = 64
image_shape = [x,y]
image_size = (x, y)

def preprocess_image(image, label):
    image = tf.image.resize(image, image_shape)        # Resize the image to 224x224
    # image = tf.image.random_flip_left_right(image)    # Random horizontal flip
    # image = tf.image.random_brightness(image, max_delta=0.1)  # Adjust brightness randomly
    image /= 255.0                                    # Normalize pixel values
    return image, label
def configure_for_performance(ds):
    ds = ds.cache()
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds

def batch_test(class_names):
    base_dir = 'Data/Data_Copy/'
    model = load_model('Models/Real/tool_classifier_model.keras')  # Adjust the path if needed

    test_dataset = keras.preprocessing.image_dataset_from_directory(
        base_dir,
        image_size=image_size,
        batch_size=50,
        label_mode='categorical'
    )

    test_dataset = test_dataset.map(lambda x, y: preprocess_image(x, y))
    test_dataset = configure_for_performance(test_dataset)

    loss, accuracy = model.evaluate(test_dataset)
    print("Test Loss:", loss)
    print("Test Accuracy:", accuracy)

    predictions = model.predict(test_dataset)
    # predicted_classes = np.argmax(predictions, axis=1)  # Assuming you're using categorical labels

    # print(predicted_classes)


def preprocess_image_for_prediction(image_path, image_size):
    image = tf.io.read_file(image_path)               # Read the image file
    image = tf.image.decode_jpeg(image, channels=3)   # Decode the image to RGB channels
    image = tf.image.resize(image, image_size)        # Resize the image
    image /= 255.0                                    # Normalize the image
    image = np.expand_dims(image, axis=0)             # Add the batch dimension
    return image

# Wrench: 'Data/Real/test/Wrench/4.png'
# CombWrench: 'Data/Real/test/CombWrench/IMG_0599.JPEG'
# Hammer: 'Data/Real/test/Hammer/IMG_0655.JPEG'
# Screwdriver: 'Data/Real/test/Screwdriver/IMG_0511.JPEG'

paths = ['Data/Real/test/Wrench/5.png', 'Data/Real/test/CombWrench/IMG_0600.JPEG', 'Data/Real/test/Hammer/IMG_0656.JPEG', 'Data/Real/test/Screwdriver/IMG_0512.JPEG']
paths_Synth = ['Data/Synthetic/Wrench/19123_img.png', 'Data/Synthetic/CombWrench/19054_img.png', 'Data/Synthetic/Hammer/19017_img.png', 'Data/Synthetic/Screwdriver/19015_img.png']
# List of classes
classes = ['Wrench', 'CombWrench', 'Hammer', 'Screwdriver']

def single_images_test(i):
    model = load_model('Models/Real/tool_classifier_model.keras')  # Adjust the path if needed
    image_path = paths_Synth[i]  # Adjust the path if needed
    image = preprocess_image_for_prediction(image_path, image_size)
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions, axis=1)  # Assuming you're using categorical labels
    print(predictions)
    return predicted_class


if __name__ == '__main__':
    # Assuming you have a list of class names in the same order as during training
    class_names_order = ['Wrench', 'CombWrench', 'Hammer', 'Screwdriver']
    class_names_predictions = ['CombWrench', 'Hammer', 'Screwdriver', 'Wrench']

    batch_test(class_names_order)
    # single_images_test()
    exit()

    for i in range(4):
        # Testing a single image
        predicted_class_index = single_images_test(i)[0]

        # Get the predicted class name
        predicted_class_name = class_names_predictions[predicted_class_index]

        # Now compare the predicted class name with the actual label
        actual_label = class_names_order[i]  # This should be the actual label for the specific test image

        print("Predicted class name:", predicted_class_name)
        print("Actual class name:", actual_label)
        print("Prediction is correct:", predicted_class_name == actual_label)