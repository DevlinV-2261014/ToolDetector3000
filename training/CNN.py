import pandas as pd
import numpy as np
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

# Load the CSV file(s)
# df = pd.read_csv('Data/Batches/CombWrench/batch_1.csv')

# Define the shape of the image
image_shape = (512, 512)

# display_first_5_images(df, image_shape)

