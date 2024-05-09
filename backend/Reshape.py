from PIL import Image
import matplotlib.pyplot as plt
img = Image.open('Data/Real/Screwdriver/IMG_0520.jpeg')
# Function to display an image
# Reshape the image
reshaped_img = img.resize((64, 64))  # Resize the image to 200x200 pixels

# Display images side by side
plt.figure(figsize=(10, 5))  # Adjust the figure size as needed

# Display the original image
plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
plt.imshow(img)
plt.title('Original Image')
plt.axis('off')  # Turn off axis numbers and ticks

# Display the reshaped image
plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
plt.imshow(reshaped_img)
plt.title('Reshaped Image')
plt.axis('off')  # Turn off axis numbers and ticks

plt.show()
