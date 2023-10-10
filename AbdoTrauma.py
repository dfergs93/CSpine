import os
import pandas as pd
import torch
import numpy as np
import pydicom
import cv2
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
from model_torch import create_model_multichannel, ensemble_predict, train_model_multi, train_model_single_validation
from data_torch import *

BASE_PATH = f'/kaggle/input/rsna-2023-abdominal-trauma-detection'
train_label = pd.read_csv(f'{BASE_PATH}/train.csv')
train_series_meta = pd.read_csv(f'{BASE_PATH}/train_series_meta.csv')
train_series_meta.merge(train_label, how = 'outer', on = 'patient_id')


def load_label_data(csv_path):
    # takes a csv file path, outputs a pandas table indexed by the StudyInstanceUID
    label_data = pd.read_csv(csv_path)
    label_data.set_index('patient_id', inplace=True)
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

def process_patient_images(folder_path, resize_shape = (256,256)):
    patient_images = []
    
    for filename in os.listdir(folder_path):
        # Load the DICOM image
        ds = pydicom.dcmread(os.path.join(folder_path, filename))

        # Normalize the pixel array
        pixel_array = ds.pixel_array * ds.RescaleSlope + ds.RescaleIntercept
        # Reshape image for model input
        image = cv2.resize(pixel_array, resize_shape)
        axial_slice = ds.ImagePositionPatient[2]
        # record slice location with the image
        patient_images.append((axial_slice,image))

    patient_images.sort(reverse=True)
    sorted_images = []
    for _, image in patient_images[30:len(patient_images)-20]: # skips the first 30 and last 20 images in the axial plane
        sorted_images.append(image)
    pixel_spacing, slice_thickness = get_pixel_spacing(folder_path)
    sorted_images = resample_image(np.stack(sorted_images), slice_thickness, pixel_spacing)
    return sorted_images


def load_data():
#    Load the label data
    label_data = load_label_data(f'{BASE_PATH}/train.csv')
    dicom_data_folder = 'dicom_data'
    
    # Load the bounding box data
    bbox_data = load_label_data('train_bounding_boxes.csv')
    angle_requests = [
        (0, 80),  # Average sagittal slice
#        (135, 50),
#        (80, 60),
#        (85,60),
#        (95,60),
#        (100,60)
#        (5,55), #supplemental obliques for augmentation
        (10,60),
#        (15,55),
#        (-5,60),
        (-10,60)
#        (-15,55),
#        (90, 60), # Average coronal slice
#        (45, 50) #Add more angles and depth fractions as needed
    ]
    windows = [
    (800,350)
    ]
#    Prepare training and testing data
    num_train = 150
    num_val = 33
    train_list, val_list, test_list = load_dicom_data('dicom_data', num_train, num_val)
    # generate training AvIPs
    X_train, y_train = generate_patient_data_multiangle_parallel(
        train_list, dicom_data_folder, label_data,
        angle_requests, multichannel = False, windows = windows, is_test_set = False)
    np.savez('Tvt0-10-10angles.npz', X_train=X_train, y_train=y_train, angle_requests = angle_requests, windows = windows)

    X_val, y_val = generate_patient_data_multiangle(val_list, dicom_data_folder, label_data, angle_requests, multichannel = False, windows = windows, is_test_set = True)
    np.savez('Tvt0-10-10angles.npz', X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)

    X_test, y_test = generate_patient_data_multiangle(test_list, dicom_data_folder, label_data, angle_requests, multichannel = False, windows = windows, is_test_set = True)
    np.savez('Tvt0-10-10angles.npz', X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test)

    print('Done')

load_data()

def preload_data(data_file):
    data = np.load(data_file)
    loaded_data = {}
    
    print("Loaded variables:")
    for var_name in data.keys():
        print(var_name)
        loaded_data[var_name] = data[var_name]
    
    return loaded_data

input_data_file = 'Tvt0-10-10angles.npz'
#
loaded_data = preload_data(input_data_file)
X_train = loaded_data['X_train']
y_train = loaded_data['y_train']
X_val = loaded_data['X_val']
y_val = loaded_data['y_val']

#X_train_w_obl = np.concatenate((X_train, X_obl), axis = 0)
#y_train_w_obl = np.concatenate((y_train, y_obl), axis = 0)

def train_model_without_kfold(X_train, y_train, X_val, y_val, model_name):
    model, optimizer, loss_fn = create_model_multichannel(X_train.shape[1])
    model = train_model_single_validation(model, optimizer, loss_fn, X_train, y_train, X_val, y_val, epochs = 12, batch_size = 32)
    torch.save(model.state_dict(), model_name)

train_model_without_kfold(X_train, y_train, X_val, y_val, 'Tvt0-10-10angles.pt')

def main_train_model(X,y,model_name, kfold_split = 3):
    print(X.shape)
    cv = StratifiedKFold(n_splits=kfold_split)
    model, optimizer, loss_fn = create_model_multichannel(X.shape[1])
    model = train_model_multi(model, optimizer, loss_fn, X, y, cv, False, epochs = 8)
    torch.save(model.state_dict(), model_name)

def main_tune_model(X,y,model_name, kfold_split = 3):
    pretrained_model = torch.load('singlechannel_model.pt')
    cv = StratifiedKFold(n_splits=kfold_split)
    model, optimizer, loss_fn = create_model_multichannel(X.shape[1])
    model.load_state_dict(pretrained_model)
    model = train_model_multi(model, optimizer, loss_fn, X, y, cv, False, epochs = 8)
    torch.save(model.state_dict(), model_name)

#train_model_name = input_data_file.split('.')[0] + '_model.pt'
#main_train_model(X_train, y_train, train_model_name)
##main_train_model(X_train_w_obl, y_train_w_obl, 'singlechannel_w_obl_model.pt')

def main_ensemble_train_model():

    # Train the ensemble of models
    cv = StratifiedKFold(n_splits=5)

    n_models = 3
    models = []
    for i in range(n_models):
        print(f"Training model {i + 3}")
        model, optimizer, loss_fn = create_model_multichannel(X_train.shape[1])
        model = train_model_multi(model, optimizer, loss_fn, X_train, y_train, cv, True)
        models.append(model)
        torch.save(model.state_dict(), f"fracture_detection_model_{i + 3}.pt")

def main_ensemble_test_model():
    # Test the ensemble of models
    ensemble_preds = ensemble_predict(models, X_test)

    # Compute and display evaluation metrics
    accuracy, precision, recall, f1, roc_auc, fpr, tpr, best_threshold = compute_metrics(y_test, ensemble_preds)
    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1:', f1)
    print('Best threshold:', best_threshold)

    # Plot the ROC curve
    plot_roc_curve(fpr,tpr, roc_auc)

