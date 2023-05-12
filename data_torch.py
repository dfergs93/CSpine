import os
import pandas as pd
import numpy as np
import pydicom
import cv2
import torch
from skimage import exposure
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, ToTensor, RandomRotation, RandomHorizontalFlip, Normalize
from scipy.ndimage import rotate
from scipy.stats import t
from concurrent.futures import ThreadPoolExecutor
import threading

print_lock = threading.Lock()

def thread_safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)
    
def load_label_data(csv_path):
    # takes a csv file path, outputs a pandas table indexed by the StudyInstanceUID
	label_data = pd.read_csv(csv_path)
	label_data.set_index('StudyInstanceUID', inplace=True)
	return label_data

def load_dicom_data(folder_path, num_train, num_val = 0):
    # takes a folder path to a folder containing multiple DICOM folders (that contain the dicom images) and makes two lists of study names
    # the nested DICOM folders are named with the StudyInstanceUID
    # if the total number of DICOM folders is n, the studies are split into a train and test set of size num_train and n - num_train
    dicom_list = os.listdir(folder_path)
    dicom_list.remove('.DS_Store')
    if num_val > 0:
        val_index = num_train+num_val
        train_list = dicom_list[:num_train]
        val_list = dicom_list[num_train:val_index]
        test_list = dicom_list[val_index:]
        return train_list, val_list, test_list
    train_list = dicom_list[:num_train]
    test_list = dicom_list[num_train:]
    return train_list, test_list

def normalize_image(ds, window = (950,400)):
    # ds is a dicom image
    # window is a tuple containing (window width, window level)
    
	ds.WindowWidth = window[0]
	ds.WindowLevel = window[1]
	pixel_array = ds.pixel_array * ds.RescaleSlope + ds.RescaleIntercept
	pixel_array[pixel_array < (ds.WindowLevel - 0.5 - (ds.WindowWidth - 1) / 2)] = 0
	pixel_array[pixel_array > (ds.WindowLevel - 0.5 + (ds.WindowWidth - 1) / 2)] = ds.WindowWidth - 1
	pixel_array = (pixel_array - (ds.WindowLevel - 0.5)) / (ds.WindowWidth - 1)
	output = pixel_array.astype(float)
	return output

def middle_slices(image, percentage=50):
    # returns the middle percentage of an axial stack
    # image is a np.stack of axial CT slices
    n_slices = image.shape[2]
    start = int((1 - percentage / 100) * n_slices / 2)
    end = n_slices - start
    return image[:, :, start:end]

def generate_train_test_data(train_list, test_list, dicom_data_folder, label_data, angle_requests = [(0,50),(90,60)]):
    # calls two instances of generate_patient_data_multiangle for the training and test set
    # not the best idea for single channel model since the test set doesn't need augmentation
    
	print('generate training set')
	X_train, y_train = generate_patient_data_multiangle(train_list, dicom_data_folder, label_data, angle_requests, multichannel = False)
	print('generate testing set')
	X_test, y_test = generate_patient_data_multiangle(test_list, dicom_data_folder, label_data, angle_requests, multichannel = False)
	return X_train, y_train, X_test, y_test

def show_avg_oblique(axial_stack, z_rotation_angle, ax):
    # display the generated AvIPs
    # z_rotation angle is a tuple (rotation angle in the z-axis, % of the axial stack to average starting in the middle)
    
    if axial_stack.any():
        # Rotate the axial stack by the specified rotation_angle (in degrees)
        rotated_axial_stack = rotate(axial_stack, z_rotation_angle[0], axes=(1, 2), reshape=False)

        # Calculate average sagittal slice from the rotated stack
        avg_sagittal = np.mean(middle_slices(rotated_axial_stack, rotation_angle[1]), axis=2)
        avg_sagittal = cv2.resize(avg_sagittal, (256, 256))
        ax.imshow(avg_sagittal, cmap="gray")
        ax.set_title(f"Angle: {rotation_angle[0]}, Depth: {rotation_angle[1]}%")
        return avg_sagittal

        
def generate_avg_oblique(axial_stack, z_rotation_angle):
    # z_rotation angle is a tuple (rotation angle in the z-axis, % of the axial stack to average starting in the middle)
    # most calls have the axial_stack as patient images, which is the output of process_patient_images
    if axial_stack.any():
        # Rotate the axial stack by the specified rotation_angle (in degrees)
        rotated_axial_stack = rotate(axial_stack, z_rotation_angle[0], axes=(1, 2), reshape=False)

        # Calculate average sagittal slice from the rotated stack
        avg_sagittal = np.mean(middle_slices(rotated_axial_stack, z_rotation_angle[1]), axis=2)
        avg_sagittal = cv2.resize(avg_sagittal, (256, 256))
        return avg_sagittal

def generate_patient_data_multiangle(patient_list, dicom_data_folder, label_data, angle_requests, multichannel = False, windows = [(950,400)], is_test_set = False):
    # takes a list of StudyInstanceUID names (patient_list) and the folder they are stored in (dicom_data_folder) and the matching label data loaded from load_label_data
    # angle_requests is a list of tuples containing (angle, % image to average over)
    # multichannel creates a single stack with multiple AvIPs to one label
    # if creating a test set, only create the first AvIP. Currently hard coded to be (0,55)
    X_data = []
    y_data = []
    if is_test_set:
    # set angle_request to only contain the sagittal
        angle_requests = [(0,100)]
        windows = [(600,100)]
    for counter, patient_id in enumerate(patient_list, 1):
    # iterate through each study folder
        print(counter, "/", len(patient_list))
        label = label_data.loc[patient_id][0]
        all_avips = []
        for window in windows:
        # for each window setting, generate the axial stack
            patient_images = process_patient_images(patient_id, dicom_data_folder, window)
            print(patient_images.shape)
            for angle in angle_requests:
            # for each angle, generate the associated AvIP
                all_avips.append(generate_avg_oblique(patient_images, angle))
        if multichannel:
            all_avips = np.stack(all_avips, axis = -1)
            X_data.append(all_avips)
            y_data.append(label)
        else:
            for avip in all_avips:
                X_data.append(avip)
                y_data.append(label)
    if multichannel:
        X_data = np.array(X_data).transpose(0, 3, 1, 2)  # Change the dimensions to (batch, channels, height, width) for PyTorch
    else:
        X_data = np.array(X_data)[:, np.newaxis, :, :]  # Add a channel dimension
    y_data = np.array(y_data).reshape(-1, 1)
    return X_data, y_data

def process_patient_images(patient_id, dicom_data_folder, window = (950,400)):
	patient_images = []
	
	for filename in os.listdir(os.path.join(dicom_data_folder, patient_id)):
		# Load the DICOM image
		ds = pydicom.dcmread(os.path.join(dicom_data_folder, patient_id, filename))
		ds.PhotometricInterpretation = 'YBR_FULL'
		# Normalize the pixel array
		image = normalize_image(ds, window)
		image = cv2.resize(image, (256, 256))
		axial_slice = ds.ImagePositionPatient[2]
		patient_images.append((axial_slice,image))
		#patient_labels.append(label)
	patient_images.sort(reverse=True)
	sorted_images = []
	for _, image in patient_images[30:len(patient_images)-20]: # skips the first 30 and last 20 images in the axial plane
		sorted_images.append(image)
	return np.stack(sorted_images)

def display_images(X_data, multichannel):
    num_images = len(X_data)

    for i in range(num_images):
        if multichannel:
            # If multichannel, display each channel separately
            num_channels = X_data[i].shape[0]
            for j in range(num_channels):
                plt.imshow(X_data[i][j], cmap='gray')
                plt.title(f'Image {i+1}, Channel {j+1}')
                plt.show()
        else:
            # If single channel, just display the image
            plt.imshow(X_data[i].squeeze(), cmap='gray')
            plt.title(f'Image {i+1}')
            plt.show()

def apply_augmentation(X, y, seed=42):
    # Define the data augmentation pipeline
    augmentation_pipeline = Compose([
        RandomRotation(degrees=(-5, 5), expand=False, center=None, fill=0),
        RandomHorizontalFlip(p=0.5)
    ])

    # Set the random seed
    torch.manual_seed(seed)

    # Apply augmentation to each sample and store the results in a list
    X_augmented = []
    y_augmented = []
    for i in range(X.shape[0]):
        x_sample = X[i]
        y_sample = y[i]

        # Permute the input data to have shape (C, H, W)
        x_sample_chw = x_sample.permute(2, 0, 1)

        # Apply the data augmentation
        x_sample_augmented = augmentation_pipeline(x_sample_chw)

        # Permute the augmented data back to shape (H, W, C)
        x_sample_augmented = x_sample_augmented.permute(1, 2, 0)

        # Append the original data and the corresponding label
        X_augmented.append(x_sample)
        y_augmented.append(y_sample)

        # Append the augmented data and the corresponding label
        X_augmented.append(x_sample_augmented)
        y_augmented.append(y_sample)

    # Stack the augmented data and labels into tensors
    X_augmented = torch.stack(X_augmented)
    y_augmented = torch.stack(y_augmented)

    return X_augmented, y_augmented

def apply_augmentation_multi(X, y, seed=42):
    # Define the data augmentation pipeline
    augmentation_pipeline = Compose([
        RandomRotation(degrees=(-5, 5), expand=False, center=None, fill=0),
        RandomHorizontalFlip(p=0.5)
    ])

    # Set the random seed
    torch.manual_seed(seed)

    # Apply augmentation to each sample and store the results in a list
    X_augmented = []
    y_augmented = []
    for i in range(X.shape[0]):
        x_sample = X[i]
        y_sample = y[i]

        # Initialize an empty list to store augmented channels
        augmented_channels = []

        # Apply the data augmentation to each channel
        for channel_idx in range(x_sample.shape[0]):
            # Permute the input data to have shape (C, H, W)
            x_sample_chw = x_sample[channel_idx].unsqueeze(0)

            # Apply the data augmentation
            x_sample_augmented = augmentation_pipeline(x_sample_chw)

            # Permute the augmented data back to shape (H, W, C)
            x_sample_augmented = x_sample_augmented.squeeze(0)

            # Add the augmented channel to the list
            augmented_channels.append(x_sample_augmented)

        # Stack the augmented channels into a single tensor
        x_sample_augmented = torch.stack(augmented_channels)

        # Append the original data and the corresponding label
        X_augmented.append(x_sample)
        y_augmented.append(y_sample)

        # Append the augmented data and the corresponding label
        X_augmented.append(x_sample_augmented)
        y_augmented.append(y_sample)

    # Stack the augmented data and labels into tensors
    X_augmented = torch.stack(X_augmented)
    y_augmented = torch.stack(y_augmented)

    return X_augmented, y_augmented

def compute_metrics(y_test, predictions, multi=False, threshold_type = 'youden'):

    if multi:
        accuracy = accuracy_score(y_test, np.round(predictions))
        precision = precision_score(y_test, np.round(predictions), average='weighted')
        recall = recall_score(y_test, np.round(predictions), average='weighted')
        f1 = f1_score(y_test, np.round(predictions), average='weighted')
        roc_auc = roc_auc_score(y_test, predictions, average='weighted', multi_class="ovr")
    if not multi:
        fpr, tpr, thresholds = roc_curve(y_test, predictions)
        roc_auc = auc(fpr, tpr)
        youden_index = tpr - fpr
        if threshold_type == 'youden':
            best_threshold = thresholds[np.argmax(youden_index)]
        # Select the threshold to maximize recall
        if threshold_type == 'max_tpr':
            best_threshold = thresholds[np.argmax(tpr)]
        binary_predicted_labels = (predictions > best_threshold).astype(int)

        # compute the confusion matrix
        cm = confusion_matrix(y_test, binary_predicted_labels)

        # compute the evaluation metrics
        accuracy = accuracy_score(y_test, binary_predicted_labels)
        precision = precision_score(y_test, binary_predicted_labels)
        recall = recall_score(y_test, binary_predicted_labels)
        f1 = f1_score(y_test, binary_predicted_labels)

        # compute sensitivity and specificity
        sensitivity = recall
        specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])

    return accuracy, precision, recall, f1, roc_auc, fpr, tpr, best_threshold, sensitivity, specificity


def plot_roc_curve(fpr, tpr, roc_auc):
	# Plot the ROC curve
	plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
	plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--', label='Random Guess')
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver Operating Characteristic (ROC) Curve')
	plt.legend(loc="lower right")
	plt.show()

def process_single_patient(patient_id, dicom_data_folder, label_data, angle_requests, multichannel, windows, is_test_set):
    thread_safe_print(patient_id)
    label = label_data.loc[patient_id][0]
    all_avips = []
    for window in windows:
        patient_images = process_patient_images(patient_id, dicom_data_folder, window)
        for angle in angle_requests:
            all_avips.append(generate_avg_oblique(patient_images, angle))

    if multichannel:
        all_avips = np.stack(all_avips, axis=-1)
        return all_avips, label
    else:
        return [(avip, label) for avip in all_avips]

def generate_patient_data_multiangle_parallel(patient_list, dicom_data_folder, label_data, angle_requests, multichannel=False, windows=[(950, 400)], is_test_set=False, n_workers=4):
    if is_test_set:
        angle_requests = [(0, 100)]
        windows = [(950, 400)]

    X_data = []
    y_data = []

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        results = executor.map(
            process_single_patient,
            patient_list,
            [dicom_data_folder] * len(patient_list),
            [label_data] * len(patient_list),
            [angle_requests] * len(patient_list),
            [multichannel] * len(patient_list),
            [windows] * len(patient_list),
            [is_test_set] * len(patient_list),
        )

        for result in results:
            if multichannel:
                X_data.append(result[0])
                y_data.append(result[1])
            else:
                for avip, label in result:
                    X_data.append(avip)
                    y_data.append(label)

    if multichannel:
        X_data = np.array(X_data).transpose(0, 3, 1, 2)
    else:
        X_data = np.array(X_data)[:, np.newaxis, :, :]

    y_data = np.array(y_data).reshape(-1, 1)

    return X_data, y_data



### No longer used ###

def generate_cs_patient_data(patient_list, dicom_data_folder, label_data):
    # Like generate_patient_data but also creates a coronal AvIP
    X_data = []
    y_data = []

    for counter, patient_id in enumerate(patient_list, 1):
        print(counter, "/", len(patient_list))
        axial_stack = process_patient_images(patient_id, dicom_data_folder)
        label = label_data.loc[patient_id]
        if axial_stack.any():
            # Calculate average sagittal slice
            avg_sagittal = np.mean(axial_stack[:, :, 2*axial_stack.shape[2] // 8:6 * axial_stack.shape[2] // 8], axis=2)
            avg_sagittal = cv2.resize(avg_sagittal, (256, 256))
                      
            # Calculate average coronal slice
            avg_coronal = np.mean(axial_stack[:, 1*axial_stack.shape[1] // 10:9 * axial_stack.shape[1] // 10, :], axis=1)
            avg_coronal = cv2.resize(avg_coronal, (256, 256))

            # Combine sagittal and coronal slices into a single array
            combined_slices = np.stack([avg_sagittal, avg_coronal], axis=-1)

            X_data.append(combined_slices)
            y_data.append(label[0])

    X_data = np.array(X_data).transpose(0, 3, 1, 2)  # Change the dimensions to (batch, channels, height, width) for PyTorch
    y_data = np.array(y_data).reshape(-1, 1)
    return X_data, y_data

