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
from data_torch import normalize_image, load_label_data, load_dicom_data, generate_train_test_data,generate_patient_data_multiangle, compute_metrics, plot_roc_curve, generate_patient_data_multiangle_parallel

def load_data():
#    Load the label data
    label_data = load_label_data('train.csv')
    dicom_data_folder = 'dicom_data'
    
    # Load the bounding box data
    bbox_data = load_label_data('train_bounding_boxes.csv')
    angle_requests = [
        (0, 10),  # Average sagittal slice
        (0, 40),
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
    (950,400),
    (600, 100),
    (10, 10)
    ]
#    Prepare training and testing data
    num_train = 150
    num_val = 33
    train_list, val_list, test_list = load_dicom_data('dicom_data', num_train, num_val)
    # generate training AvIPs
    X_train, y_train = generate_patient_data_multiangle_parallel(
        train_list, dicom_data_folder, label_data,
        angle_requests, multichannel = False, windows = windows, is_test_set = False)
    np.savez('4 obliques and 3 windows with train test and val sets.npz', X_train=X_train, y_train=y_train, angle_requests = angle_requests, windows = windows)

    X_val, y_val = generate_patient_data_multiangle(val_list, dicom_data_folder, label_data, angle_requests, multichannel = False, windows = windows, is_test_set = True)
    np.savez('4 obliques and 3 windows with train test and val sets.npz', X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)

    X_test, y_test = generate_patient_data_multiangle(test_list, dicom_data_folder, label_data, angle_requests, multichannel = False, windows = windows, is_test_set = True)
    np.savez('4 obliques and 3 windows with train test and val sets.npz', X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test)

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

input_data_file = '4 obliques and 3 windows with train test and val sets.npz'
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

train_model_without_kfold(X_train, y_train, X_val, y_val, '4o3w_model.pt')

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

