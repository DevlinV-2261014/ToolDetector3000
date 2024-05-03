from sklearn.metrics import confusion_matrix
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

#Directory
test_dir = "./data/Real/"
class_names = ['CombWrench', 'Hammer', 'Screwdriver', 'Wrench']

# Load our model
MODEL = load_model("tool_classifier_model.keras")

# Need for the labels
true_labels = []
predicted_labels = []

for class_name in class_names: # For each folder of the classes
    class_dir = os.path.join(test_dir, class_name) # Get the path of the class
    for file in os.listdir(class_dir): # Go over every image in here
        img_path = os.path.join(class_dir, file)
        img = image.load_img(img_path, target_size=(64, 64))
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0
        img_batch = np.expand_dims(img_array, axis=0)

        prediction = MODEL.predict(img_batch)
        predicted_class = class_names[np.argmax(prediction)] # Get the class with the highest probability

        true_labels.append(class_name) # Add the true label
        predicted_labels.append(predicted_class) # Add the predicted label

cm = confusion_matrix(true_labels, predicted_labels, labels=class_names) # Make the confusion matrix

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
