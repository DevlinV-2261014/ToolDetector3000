import os
from PIL import Image, ImageOps
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Iterate over DataFrame rows
def display_first_5_images(df, image_shape):
    for idx, row in df.iterrows():
        # Extract label
        label = row['Label']
        # Extract pixel data: skip 'Label' column by selecting all columns after 'Label'
        pixel_data = row[1:].values  # Assuming 'Label' is the first column
        pixel_data = pixel_data.astype(np.uint8)  # Ensure data type is correct for image display

        # Reshape pixel data into an image
        img = pixel_data.reshape(image_shape)

        # Display the image
        plt.imshow(img, cmap='gray')  # Use 'gray' for grayscale images
        plt.title(f"Label: {label}")
        plt.show()

        # Limit to the first 5 images
        if idx == 5:
            break

def save_batches_to_csv(data, batch_size, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

    for key, df in data.items():
        # Splitting the dataframe into batches
        num_batches = (len(df) + batch_size - 1) // batch_size  # Calculate how many batches are needed
        for i in range(num_batches):
            batch_df = df[i * batch_size:(i + 1) * batch_size]
            batch_df.to_csv(f"Data/{directory}/{key}/batch_{i + 1}.csv", index=False)

def load_images_from_folder(folder, numeric_label, size=(512, 512)):
    """ Load images from a directory, convert them to grayscale, resize them to 512x512, and return a DataFrame with numeric labels and flattened image data. """
    data = []
    count = 0
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        try:
            with Image.open(img_path) as img:
                # to grayscale and resize
                img = img.convert('L').resize(size, Image.Resampling.LANCZOS)
                img_data = np.array(img).flatten()
                # Create a complete row with the label and image data
                row = np.concatenate([[numeric_label], img_data])
                data.append(row)
                count += 1
        except IOError:
            print(f"Error opening or reading image {filename}")
        print(count)

    if data:
        data = np.array(data)
        num_cols = data.shape[1]  # Get the total number of columns
        column_names = ['Label'] + [f'Pixel{i}' for i in range(num_cols - 1)]
        return pd.DataFrame(data, columns=column_names)
    else:
        return pd.DataFrame()  # Return an empty DataFrame if no data was loaded


def pad_images_to_size(folder_path, target_size=(1773, 1773)):
    padded_count = 0  # Counter for how many images were padded

    # Iterate over all files in the specified directory
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        try:
            # Open the image
            with Image.open(file_path) as img:
                # Check if the image needs padding
                if img.size != target_size:
                    # Get the size of the image
                    original_size = img.size

                    # Calculate padding
                    delta_width = target_size[0] - original_size[0]
                    delta_height = target_size[1] - original_size[1]
                    padding = (delta_width // 2, delta_height // 2, delta_width - (delta_width // 2),
                               delta_height - (delta_height // 2))

                    # Pad the image and center it
                    new_img = ImageOps.expand(img, padding)
                    # Save the padded image back to disk
                    new_img.save(file_path)
                    padded_count += 1
        except IOError:
            print(f"Error opening or processing file {filename}")

    return padded_count


def remove_images_of_different_size(folder_path, target_size=(1773, 1773)):
    images_removed = 0  # Counter to keep track of how many images were removed
    folder_path = Path(folder_path)  # Ensure folder_path is a Path object

    # List all files in the directory
    for file_path in folder_path.iterdir():
        try:
            # Open the image file
            with Image.open(file_path) as img:
                # Check if the image size matches the target size
                if img.size != target_size:
                    file_path.unlink()  # Use unlink method from pathlib
                    images_removed += 1  # Increment the counter
        except IOError:
            # Handle the error if the file could not be opened
            print(f"Error opening or processing file {file_path.name}")

    return images_removed

# Base path to your image folders
base_path = "Data/Real"

# Mapping from folder names to numeric labels
folders = {
    "Wrench": 0,
    "Screwdriver": 1,
    "Hammer": 2,
    "CombWrench": 3
}

# Check if all images are of the same size: this should be in order
for folder in folders:
    folder_path = os.path.join(base_path, folder)
    images_removed = remove_images_of_different_size(folder_path)
    print(f"Removed {images_removed} images from {folder}")

# Load images from each folder into separate DataFrames
data = {}
for folder, label in folders.items():
    # path = base_path + "/" + folder
    path = base_path + "/" + folder + "_Augmentation"
    df = load_images_from_folder(path, label)
    data[folder] =  df

# Define the shape of the image (e.g., 28x28 for MNIST)
image_shape = (512, 512)

display_first_5_images(data["Wrench"], image_shape)
display_first_5_images(data["Screwdriver"], image_shape)
display_first_5_images(data["Hammer"], image_shape)
display_first_5_images(data["CombWrench"], image_shape)

# Loop per key-value pair in all_data
save_batches_to_csv(data, batch_size=50, directory='Batches')

