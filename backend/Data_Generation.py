import os
from keras_preprocessing.image import ImageDataGenerator, load_img, img_to_array

def save_augmented_images(source_directory, target_directory, augment_count=5):
    # Define your image data generator with augmentations
    data_gen = ImageDataGenerator(
        rotation_range=180,
        # width_shift_range=0.2,
        # height_shift_range=0.2,
        # shear_range=0.2,
        # zoom_range=0.2,
        horizontal_flip=True
    )

    # Create target directory if it doesn't exist
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    # Process each image in the source directory
    for filename in os.listdir(source_directory):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".JPEG"):  # check for image files
            img_path = os.path.join(source_directory, filename)
            img = load_img(img_path)  # Load image
            img_array = img_to_array(img)  # Convert it to array
            img_array = img_array.reshape((1,) + img_array.shape)  # Reshape it to (1, height, width, channels)

            # Generate 'augment_count' new images per original image
            i = 0
            for batch in data_gen.flow(img_array, batch_size=1,
                                       save_to_dir=target_directory, save_prefix='aug', save_format='jpeg'):
                i += 1
                if i >= augment_count:
                    break  # Break the loop to ensure we only augment 'augment_count' times


# Example usage
folder_to_label = {
    "Wrench": 0,
    "Screwdriver": 1,
    "Hammer": 2,
    "CombWrench": 3
}

for folder in folder_to_label:
    source_dir = 'Data/Real' + "/" + folder
    target_dir = source_dir + "_Augmentation"
    save_augmented_images(source_dir, target_dir, augment_count=20)
