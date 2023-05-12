import os
import cv2
import numpy as np
import torch
from evaluate_torch import evaluate_model 

def load_and_preprocess_image(image_path, target_size):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print(f"Error reading image: {image_path}")
        return None
    
    image = cv2.resize(image, target_size)
    image = image / 255.0  # Normalize pixel values to the range [0, 1]
    return image


def load_test_images(folder_path_tuple, target_size=(256, 256)):
    image_paths = [os.path.join(folder_path_tuple[0], f) for f in os.listdir(folder_path_tuple[0]) if f.lower().endswith(('.jpeg', '.jpg')) and not f.startswith('.')]
    test_images = []
    test_labels = []
    label = folder_path_tuple[1]
    for image_path in image_paths:
        image = load_and_preprocess_image(image_path, target_size)
        test_images.append(image)
        test_labels.append(label)
    X_data = np.array(test_images)[:, np.newaxis, :, :]
    y_data = np.array(test_labels).reshape(-1, 1)
    return X_data, y_data

# Load and preprocess test images
folder_path_list = [("./KaggleCSpineXR/c-spine_normal",1),("./KaggleCSpineXR/c-spine_fracture",0),("./KaggleCSpineXR/c-spine_dislocation",0)]

X_jpeg_test = []
y_jpeg_test = []

for folder in folder_path_list:
    X, y = load_test_images(folder)
    X_jpeg_test.append(X)
    y_jpeg_test.append(y)

X_jpeg_test = np.concatenate(X_jpeg_test, axis=0)
y_jpeg_test = np.concatenate(y_jpeg_test, axis=0)

evaluate_model(X_jpeg_test,y_jpeg_test,'singlechannel_supp_obliques_2_model.pt')

