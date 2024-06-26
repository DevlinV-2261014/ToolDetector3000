import cv2
import numpy as np
import os
from skimage.restoration import estimate_sigma

def estimate_noise_level(image, useEstimateSigma):
    # Convert the image to grayscale if it's not already
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    if (not useEstimateSigma):
        # Calculate the variance of pixel intensities as an estimate of noise
        # Higher number means more noise
        noise_level = np.var(gray_image)
        return noise_level
    noise_level = estimate_sigma(gray_image, average_sigmas=True, channel_axis=-1) # Lower value = More noise
    return noise_level

tools = ['CombWrench', 'Hammer', 'Screwdriver', 'Wrench'] #The tools
folder_path = 'Data/Real/' #The folder path

# Just some print that has some info
for tool in tools: # For every tool
    current_path = f'{folder_path}{tool}' #Change path
    # List all files in the folder
    image_files = os.listdir(current_path)

    count = 0 # We need this to calculate the average noise level

    # Iterate over each image file
    for filename in image_files:
        # Load the image
        image_path = os.path.join(current_path, filename)
        image = cv2.imread(image_path)
        down_width = 212
        down_height = 212
        down_points = (down_width, down_height)
        resized_down = cv2.resize(image, down_points, interpolation= cv2.INTER_LINEAR)

        # Estimate noise level
        noise_level = estimate_noise_level(resized_down, True) # Base it off the function
        count = count + noise_level
       
        #print("Noise level of", filename, ":", noise_level)

    print("Average noise level for", tool, ":" , round(count / len(image_files), 3))
