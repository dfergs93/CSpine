import os
import pandas as pd
import torch.nn as nn
from torch.optim import Adam
from sklearn.model_selection import StratifiedKFold
import timm
import random
from data_torch import apply_augmentation
import torch
import numpy as np
import pydicom
import cv2
from sklearn.model_selection import StratifiedKFold, KFold
from scipy.ndimage import rotate
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from model_torch import create_2_model, ensemble_predict, train_model, create_model_multichannel, train_model_multi, single_predict
from data_torch import normalize_image, load_label_data, load_dicom_data, generate_patient_data_multiangle, compute_metrics, plot_roc_curve, apply_augmentation, show_avg_oblique, display_images, generate_avg_oblique, middle_slices
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageOps

default_folder_path = '/Users/duncanferguson/Desktop/CSpine/dicom_data/1.2.826.0.1.3680043.634'

def process_patient_images(folder_path, window = (950,400)):
    patient_images = []
    
    for filename in os.listdir(folder_path):
        # Load the DICOM image
        ds = pydicom.dcmread(os.path.join(folder_path, filename))
        ds.PhotometricInterpretation = 'YBR_FULL'
        # Normalize the pixel array
        image = normalize_image(ds, window)
        
        axial_slice = ds.ImagePositionPatient[2]
        patient_images.append((axial_slice,image))
        #patient_labels.append(label)
    patient_images.sort(reverse=True)
    sorted_images = []
    for _, image in patient_images[30:len(patient_images)-20]: # skips the first 30 and last 20 images in the axial plane
        sorted_images.append(image)
    return np.stack(sorted_images)

def middle_slices_2(image, points):
    # returns the middle percentage of an axial stack
    # image is a np.stack of axial CT slices
    x_mean = int((points[1][0]+points[0][0])/2)
#    y_mean = int((points[1][1]+points[0][1])/2)
    z_mean = int((points[1][2]+points[0][2])/2)
    
    # Ensure coordinates are within image bounds
    x_min = max(0, x_mean-150)
    x_max = min(image.shape[0], x_mean+150)
#    y_min = max(0, y_mean-150)
#    y_max = min(image.shape[1], y_mean+150)
    z_min = max(0, z_mean-150)
    z_max = min(image.shape[2], z_mean+150)
    
    return image[x_min:x_max,z_min:z_max,:]


def generate_avg_oblique_2(axial_stack, x_angle, z_angle, points):
    axial_stack = middle_slices_2(axial_stack, points)
    # Rotate the axial stack by the specified rotation_angle (in degrees) in the x-axis
    rotated_axial_stack = rotate(axial_stack, x_angle, axes=(0, 1), reshape=False)
    # Rotate the axial stack by the specified rotation_angle (in degrees) in the z-axis
    rotated_axial_stack = rotate(rotated_axial_stack, z_angle, axes=(1, 2), reshape=False)

    # Calculate average sagittal slice from the rotated stack
    avg_sagittal = np.mean(rotated_axial_stack, axis=2)
    return avg_sagittal


def generate_single_patient_image(folder_path, angle, window):

    patient_id = os.path.basename(folder_path)
    # Generate the axial stack
    patient_images = process_patient_images(folder_path, window)

    avip = generate_avg_oblique(patient_images, angle)

    return avip

def display_generated_image(image):
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.show()

def scroll_axial_stack(axial_stack):
    fig, ax = plt.subplots()
    slice_index = len(axial_stack) // 2
    slider_ax = plt.axes([0.25, 0.1, 0.65, 0.03])

    points = []

    def update(val):
        nonlocal slice_index
        slice_index = int(val)
        ax.clear()
        ax.imshow(axial_stack[slice_index], cmap='gray')
        for point in points:
            if point[0] == slice_index:
                ax.plot(point[2], point[1], 'ro')
        fig.canvas.draw_idle()

    def onclick(event):
        nonlocal slice_index
        if event.inaxes == ax:
            x, y = event.xdata, event.ydata
            points.append((x, y, slice_index))

            if len(points) == 2:
                x_angle, z_angle = calculate_required_angles(points)
                print("Required angles: x =", x_angle, "z =", round(z_angle-90.0,1))

                # Call generate_avg_oblique with the calculated x_angle and z_angle
                avg_oblique_image = generate_avg_oblique_2(axial_stack, x_angle, z_angle, points)

                # Display the generated image using the display_generated_image function in a non-blocking way
                plt.ion()
                display_generated_image(avg_oblique_image)
                plt.pause(0.001)

                # Clear the points list for the next two points
                points.clear()

            ax.clear()
            ax.imshow(axial_stack[slice_index], cmap="gray")
            for x, y, z in points:
                if z == slice_index:
                    ax.plot(x, y, "ro")
            fig.canvas.draw()
    def onscroll(event):
        nonlocal slice_index
        if event.button == 'up':
            slice_index += 1
        elif event.button == 'down':
            slice_index -= 1

        # Ensure slice_index stays within the valid range
        slice_index = min(max(slice_index, 0), len(axial_stack) - 1)

        # Update the displayed slice and the slider value
        ax.clear()
        ax.imshow(axial_stack[slice_index], cmap='gray')
        slider.set_val(slice_index)
        fig.canvas.draw_idle()

    slider = Slider(slider_ax, 'Slice', 0, len(axial_stack) - 1, valinit=slice_index, valstep=1)
    slider.on_changed(update)

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    cid = fig.canvas.mpl_connect('scroll_event', onscroll)


    ax.imshow(axial_stack[slice_index], cmap='gray')
    plt.show()


def calculate_required_angles(points, axial_spacing=1):
    dx = points[1][0] - points[0][0]
    dy = points[1][1] - points[0][1]
    dz = (points[1][2] - points[0][2]) * axial_spacing

    x_angle = np.arctan2(dy, dz) * 180 / np.pi
    x_angle = round(90.0 - x_angle, 1)
    z_angle = round(np.arctan2(dy, dx) * 180 / np.pi, 1)

    return x_angle, z_angle



#folder_path = 'dicom_data/1.2.826.0.1.3680043.3749'
#angle = (0, 100)  # Example angle requests
#window = (950, 400)  # Example window

#image = generate_single_patient_image(folder_path, angle_requests, window)

def create_image():
    angle_input = angle_var.get()
    window_input = window_var.get()
    folder = folder_var.get()

    try:
        angle_requests = [tuple(map(int, angle_input.split(',')))]
        windows = [tuple(map(int, window_input.split(',')))]
    except ValueError:
        print("Invalid input format for angles or windows")
        return

    image = generate_single_patient_image(folder, angle_requests[0], windows[0])

    # Display the generated image using pyplot
    plt.imshow(image, cmap='gray')
    plt.show()

def display_image(image):
    image = Image.fromarray((image * 255).astype('uint8'), 'L')
    image = ImageOps.equalize(image)
    image = image.resize((256, 256), Image.LANCZOS)
    photo = ImageTk.PhotoImage(image)

    label.config(image=photo)
    label.image = photo

def browse_folder():
    folder_selected = filedialog.askdirectory()
    folder_var.set(folder_selected)

root = tk.Tk()

angle_var = tk.StringVar(value = '0,80')
window_var = tk.StringVar(value = '950,400')
folder_var = tk.StringVar(value=default_folder_path)

scroll_button = tk.Button(root, text="Scroll Axial Stack", command=lambda: scroll_axial_stack(process_patient_images(folder_var.get())))
scroll_button.grid(row=5, column=0, columnspan=3)

angle_label = tk.Label(root, text="Angle (angle, %):")
angle_entry = tk.Entry(root, textvariable=angle_var)
window_label = tk.Label(root, text="Window (width, height):")
window_entry = tk.Entry(root, textvariable=window_var)
folder_label = tk.Label(root, text="DICOM Folder:")
folder_entry = tk.Entry(root, textvariable=folder_var)
folder_button = tk.Button(root, text="Browse", command=browse_folder)
generate_button = tk.Button(root, text="Generate Image", command=create_image)
label = tk.Label(root)

angle_label.grid(row=0, column=0)
angle_entry.grid(row=0, column=1)
window_label.grid(row=1, column=0)
window_entry.grid(row=1, column=1)
folder_label.grid(row=2, column=0)
folder_entry.grid(row=2, column=1)
folder_button.grid(row=2, column=2)
generate_button.grid(row=3, column=0, columnspan=3)
label.grid(row=4, column=0, columnspan=3)

root.mainloop()
    
#display_images(x, multichannel = False)
