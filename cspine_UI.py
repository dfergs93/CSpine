import os
import pandas as pd
import numpy as np
import pydicom
import cv2
from scipy.ndimage import rotate, zoom
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageOps

default_folder_path = '/Users/duncanferguson/Desktop/CSpine/dicom_data/1.2.826.0.1.3680043.634'
output_path = '/Users/duncanferguson/Desktop/CSpine/npz_datasets'

def process_patient_images(folder_path):
    patient_images = []
    patient_id = folder_path.split("/")[-1]
    for filename in os.listdir(folder_path):
        # Load the DICOM image
        ds = pydicom.dcmread(os.path.join(folder_path, filename))
        # Normalize the pixel array
        pixel_array = ds.pixel_array * ds.RescaleSlope + ds.RescaleIntercept
        axial_slice = ds.ImagePositionPatient[2]
        patient_images.append((axial_slice, pixel_array))
    
    patient_images.sort(reverse=True)
    sorted_images = []
    for _, image in patient_images: # skips the first 30 and last 20 images in the axial plane
        sorted_images.append(image)
    np.save(output_path + f"{patient_id}.npy", np.stack(sorted_images))
    return np.stack(sorted_images)
    
def load_plane(axial_stack, plane = 'sag'):
    global global_axial_stack
    if plane == 'sag':
        loaded_stack = axial_stack.transpose(2,0,1)
    elif plane == 'cor':
        loaded_stack = axial_stack.transpose(1,0,2)
    else:
        raise ValueError("Invalid plane variable. Accepted variables are sag or cor")
    return loaded_stack

def middle_slices(image, percentage=50):
    # returns the middle percentage of an axial stack
    # image is a np.stack of axial CT slices
    n_slices = image.shape[2]
    start = int((1 - percentage / 100) * n_slices / 2)
    end = n_slices - start
    return image[:, :, start:end]

def apply_window(image_stack, window, rescale = False):
    window_width = window[0]
    window_center = window[1]
    img = image_stack.copy()
    img_min = window_center - window_width//2 #minimum HU level
    img_max = window_center + window_width//2 #maximum HU level
    img[img<img_min] = img_min #set img_min for all HU levels less than minimum HU level
    img[img>img_max] = img_max #set img_max for all HU levels higher than maximum HU level
    if rescale:
        img = (img - img_min) / (img_max - img_min)*255.0
    return img.astype(float)

def apply_mask(image_stack, window, mask_window = (700,400)):
    mask = mask_patient_images(image_stack, mask_window)
    windowed_stack = apply_window(image_stack, window)
    windowed_stack = windowed_stack * mask
    return windowed_stack

def create_mask(image, threshold=150):
    """
    Create a binary mask where all pixels above a certain threshold are kept.
    """
    mask = image > threshold
    mask = np.not_logical(mask)
    return mask

def mask_patient_images(axial_stack, window = (700,400)):
    # Create a 3D mask from the normalized image stack
    axial_stack = apply_window(axial_stack, window)
    mask = create_mask(axial_stack)
    return mask

def center_stack(image_stack, point, num_slices=400):
    half_slices = num_slices // 2
    x, y, center_index = point
    x, y, center_index = int(x), int(y), int(center_index)

    lower_index = max(0, center_index - half_slices)
    upper_index = min(image_stack.shape[0], center_index + half_slices)

    # Determine the x and y bounds for the region of interest
    lower_x = max(0, x - half_slices)
    upper_x = min(image_stack.shape[2], x + half_slices)
    lower_y = max(0, y - half_slices)
    upper_y = min(image_stack.shape[1], y + half_slices)

    # Return the region of interest
    return image_stack[lower_index:upper_index, lower_y:upper_y, lower_x:upper_x]

def resample_image(image_stack, slice_thickness, pixel_spacing):
    # Calculate the current pixel dimensions
    x_pixel_dim, y_pixel_dim = pixel_spacing
    z_pixel_dim = slice_thickness

    # Calculate the scaling factors
    x_scale = x_pixel_dim / min([x_pixel_dim, y_pixel_dim, z_pixel_dim])
    y_scale = y_pixel_dim / min([x_pixel_dim, y_pixel_dim, z_pixel_dim])
    z_scale = z_pixel_dim / min([x_pixel_dim, y_pixel_dim, z_pixel_dim])

    # Rescale the image stack
    resampled_image_stack = zoom(image_stack, (z_scale, y_scale, x_scale))

    return resampled_image_stack

def generate_avg_oblique_2(axial_stack, x_angle, z_angle):
    axial_stack = middle_slices(axial_stack, 80)
    # Rotate the axial stack by the specified rotation_angle (in degrees) in the x-axis
    rotated_axial_stack = rotate(axial_stack, x_angle, axes=(0, 1), reshape=False)
    # Rotate the axial stack by the specified rotation_angle (in degrees) in the z-axis
    rotated_axial_stack = rotate(rotated_axial_stack, z_angle, axes=(1, 2), reshape=False)

    # Calculate average sagittal slice from the rotated stack
#    avg_sagittal = np.mean(rotated_axial_stack, axis=2)
    avg_sagittal = weighted_average_3d(rotated_axial_stack, 0.8, 0.2)
    return avg_sagittal

def weighted_average_3d(image_stack, start_weight=0.5, end_weight=0.5):
    """
    Compute the weighted average along the third axis of a 3D image stack.

    Parameters:
    - image_stack: The input 3D image stack (height, width, depth)
    - start_weight: The weight for the first slice (default 0.5)
    - end_weight: The weight for the last slice (default 0.5)

    Returns: A 2D numpy array representing the weighted average image
    """
    # Create a 1D weight vector of the same length as the third dimension of the image stack
    weight_vector = np.linspace(start_weight, end_weight, image_stack.shape[2])

    # Expand the weight vector into a 3D weight tensor of the same shape as the image stack
    weight_tensor = np.repeat(weight_vector[np.newaxis, np.newaxis, :], image_stack.shape[0], axis=0)
    weight_tensor = np.repeat(weight_tensor, image_stack.shape[1], axis=1)

    # Compute the weighted average along the third axis
    weighted_average = np.sum(image_stack * weight_tensor, axis=2) / np.sum(weight_tensor, axis=2)

    return weighted_average

def generate_single_patient_image(axial_stack, angle, window, mask = False):
    windowed_stack = apply_window(axial_stack, window)
    # Generate the axial stack
    if mask:
        windowed_stack = apply_mask(axial_stack,window)
    
    avip = generate_avg_oblique_2(windowed_stack, 0, angle)

    return avip

def display_generated_image(image):
    plt.figure()
    plt.imshow(image, cmap='gray')
    mid_y, mid_x = np.array(image.shape) // 2  # Calculate the center point of the image
    plt.plot(mid_x, mid_y, 'w,')  # 'ro' stands for red color and circle marker
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
                ax.plot(point[2], point[1], 'r.')
        fig.canvas.draw_idle()

    def onclick(event):
        nonlocal slice_index
        if event.inaxes == ax:
            x, y = event.xdata, event.ydata
            points.append((x, y, slice_index))

            if len(points) == 2:
                x_angle, z_angle = calculate_required_angles(points)
                print("Required angles: CC =", x_angle, "TR =", -1* round(z_angle-270.0,1))
                num_slices = int(center_var.get())
                centered_stack = center_stack(axial_stack, points[0], num_slices)
                # Call generate_avg_oblique with the calculated x_angle and z_angle
                avg_oblique_image = generate_avg_oblique_2(centered_stack, x_angle, z_angle)

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
                    ax.plot(x, y, "r.")
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

def scroll_sagittal_stack(axial_stack):
    sagittal_stack = load_plane(axial_stack, "sag")
    fig, ax = plt.subplots()
    slice_index = len(sagittal_stack) // 2
    slider_ax = plt.axes([0.25, 0.1, 0.65, 0.03])

    points = []

    def update(val):
        nonlocal slice_index
        slice_index = int(val)
        ax.clear()
        ax.imshow(sagittal_stack[slice_index], cmap='gray')
        for point in points:
            if point[0] == slice_index:
                ax.plot(point[2], point[1], 'ro')
        fig.canvas.draw_idle()

    def onclick(event):
        nonlocal slice_index
        if event.inaxes == ax:
            x, y = event.xdata, event.ydata
            points.append((slice_index,x,y))

            if len(points) == 2:
                # Draw the line between the two points
                ax.plot([points[0][1], points[1][1]], [points[0][2], points[1][2]], 'r-')
                ax.figure.canvas.draw()
                answer = input("Accept this line? (yes/no): ")
                if answer == "yes":
                    x_angle, z_angle = calculate_required_angles(points)
                    print("Required angles: CC =", x_angle, "TR =", -1* round(z_angle-270.0,1))
                    num_slices = int(center_var.get())
                    centered_stack = center_stack(axial_stack, points[1], num_slices)
                    # Call generate_avg_oblique with the calculated x_angle and z_angle
                    avg_oblique_image = generate_avg_oblique_2(centered_stack, x_angle, z_angle)

                    # Display the generated image using the display_generated_image function in a non-blocking way
                    plt.ion()
                    display_generated_image(avg_oblique_image)
                    plt.pause(0.001)
                else:
                    print("Please redraw the line")
                # Clear the points list for the next two points
                points.clear()

            ax.clear()
            ax.imshow(sagittal_stack[slice_index], cmap="gray")
            for z,x,y in points:
                if z == slice_index:
                    ax.plot(x, y, "r.")
            fig.canvas.draw()
    def onscroll(event):
        nonlocal slice_index
        if event.button == 'up':
            slice_index += 1
        elif event.button == 'down':
            slice_index -= 1

        # Ensure slice_index stays within the valid range
        slice_index = min(max(slice_index, 0), len(sagittal_stack) - 1)

        # Update the displayed slice and the slider value
        ax.clear()
        ax.imshow(sagittal_stack[slice_index], cmap='gray')
        slider.set_val(slice_index)
        fig.canvas.draw_idle()

    slider = Slider(slider_ax, 'Slice', 0, len(sagittal_stack) - 1, valinit=slice_index, valstep=1)
    slider.on_changed(update)

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    cid = fig.canvas.mpl_connect('scroll_event', onscroll)


    ax.imshow(sagittal_stack[slice_index], cmap='gray')
    plt.show()


def calculate_required_angles(points, axial_spacing=1):
    dx = points[1][0] - points[0][0]
    dy = points[1][1] - points[0][1]
    dz = (points[1][2] - points[0][2]) * axial_spacing

    x_angle = np.arctan2(dy, dz) * 180 / np.pi
    x_angle = round(90.0 - x_angle, 1)
    z_angle = round(np.arctan2(dy, dx) * 180 / np.pi, 1)+180

    return x_angle, z_angle


#squished image /Users/duncanferguson/Desktop/CSpine/dicom_data/1.2.826.0.1.3680043.258
#folder_path = 'dicom_data/1.2.826.0.1.3680043.3749'
#print(process_patient_images(folder_path))
#angle = (0, 100)  # Example angle requests
#window = (950, 400)  # Example window

#image = generate_single_patient_image(folder_path, angle_requests, window)


def create_image(mask = False):
    angle_input = angle_var.get()
    window_input = window_var.get()

    try:
        angle_requests = [tuple(map(int, angle_input.split(',')))]
        windows = [tuple(map(int, window_input.split(',')))]
    except ValueError:
        print("Invalid input format for angles or windows")
        return

    image = generate_single_patient_image(global_axial_stack, angle_requests[0], windows[0], mask)

    # Display the generated image using pyplot
    plt.imshow(image, cmap='gray')
    plt.show()

def browse_folder():
    folder_selected = filedialog.askdirectory()
    folder_var.set(folder_selected)
    load_axial_stack()

def load_axial_stack():
    global global_axial_stack
    folder_path = folder_var.get()
    global_axial_stack = process_patient_images(folder_path)
    pixel_spacing, slice_thickness = get_pixel_spacing(folder_path)
    global_axial_stack = resample_image(global_axial_stack, slice_thickness, pixel_spacing)
    print('Done Loading')

def get_pixel_spacing(folder_path):
    first_file = os.listdir(folder_path)[0]
    ds = pydicom.dcmread(os.path.join(folder_path, first_file))
#    const_pixel_dims = (int(ds.Rows), int(ds.Columns), len(os.listdir(folder_path)))
    return (float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1])),float(ds.SliceThickness)
    
root = tk.Tk()

angle_var = tk.StringVar(value = '0,80')
window_var = tk.StringVar(value = '950,400')
folder_var = tk.StringVar(value=default_folder_path)
center_var = tk.StringVar(value = '300')
load_axial_stack()


angle_label = tk.Label(root, text="Angle (angle, %):")
angle_entry = tk.Entry(root, textvariable=angle_var)
window_label = tk.Label(root, text="Window (width, height):")
window_entry = tk.Entry(root, textvariable=window_var)
center_label = tk.Label(root, text="Center Size:")
center_entry = tk.Entry(root, textvariable=center_var)
folder_label = tk.Label(root, text="DICOM Folder:")
folder_entry = tk.Entry(root, textvariable=folder_var)
folder_button = tk.Button(root, text="Browse", command=browse_folder)

generate_button = tk.Button(root, text="Generate Image", command=create_image)
mask_button = tk.Button(root, text="Generate Mask Image", command=lambda:create_image(mask=True))
label = tk.Label(root)
scroll_button = tk.Button(root, text="Scroll Axial Stack", command=lambda: scroll_axial_stack(apply_window(global_axial_stack, window = tuple(map(int, window_var.get().split(','))))))
scroll_button.grid(row=5, column=0, columnspan=3)
sag_scroll_button = tk.Button(root, text="Scroll Sagittal Stack", command=lambda: scroll_sagittal_stack(apply_window(global_axial_stack, window = tuple(map(int, window_var.get().split(','))))))
sag_scroll_button.grid(row=5, column=1, columnspan=3)
scroll_button2 = tk.Button(root, text="Scroll Masked Stack", command=lambda: scroll_axial_stack(apply_mask(global_axial_stack, window = tuple(map(int, window_var.get().split(','))))))
scroll_button2.grid(row=6, column=0, columnspan=3)


angle_label.grid(row=0, column=0)
angle_entry.grid(row=0, column=1)
window_label.grid(row=1, column=0)
window_entry.grid(row=1, column=1)
center_label.grid(row=1, column=2)
center_entry.grid(row=1, column=3)
folder_label.grid(row=2, column=0)
folder_entry.grid(row=2, column=1)
folder_button.grid(row=2, column=2)
generate_button.grid(row=3, column=0)
mask_button.grid(row=3, column=1)
label.grid(row=7, column=0, columnspan=3)

root.mainloop()

