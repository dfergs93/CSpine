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
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
from model_torch import create_2_model, ensemble_predict, train_model, create_model_multichannel, train_model_multi, single_predict
from data_torch import normalize_image, load_label_data, load_dicom_data, generate_patient_data_multiangle, compute_metrics, plot_roc_curve, apply_augmentation, show_avg_oblique, display_images

def load_test_data():
    # Load the label data
    label_data = load_label_data('train.csv')
    num_train = 232
    train_list, test_list = load_dicom_data('dicom_data', num_train)
    # #
    slice_requests = [
    #  (angle_rotation in z-axis, depth start, depth end)
    #  0 = sagittal, 90 = coronal
    (0, 10),
    (0, 40),
    (0, 100),
    (90, 80)
#    (45, 50),
#    (90, 60)
    # Add more angles and depth fractions as needed
    ]
    windows = [
    (950,400),
    (600, 100),
#    (10, 10)
    ]
    print(test_list)
    X_test, y_test = generate_patient_data_multiangle(test_list, 'dicom_data', label_data, slice_requests, multichannel = False, windows = windows, is_test_set = False)
    return X_test, y_test

x,y = load_test_data()

def preload_test_data(test_data_file, x = 'X_test', y = 'y_test'):
    data = np.load(test_data_file)
    return data[x], data[y]

#X_train, y_train = preload_test_data('data2.npz','X_train', 'y_train')
#
#X_test, y_test = preload_test_data('data3.npz')

def show_obliques():
    fig, axes = plt.subplots(1, len(slice_requests), figsize=(25, 5))

    for i, angle in enumerate(slice_requests):
        show_avg_oblique(test_list[1], 'dicom_data', angle, axes[i])

    plt.tight_layout()
    plt.show()

display_images(x, multichannel = False)
#print(X_train.shape)

def evaluate_model(X_test, y_test, model_name):
    model, optimizer, loss_fn = create_model_multichannel(X_test.shape[1])
    model.load_state_dict(torch.load(model_name))
    model.eval()
    prediction = single_predict(model, X_test)
    # Compute and display evaluation metrics
    accuracy, precision, recall, f1, roc_auc, fpr, tpr, best_threshold, sensitivity, specificity = compute_metrics(y_test, prediction, threshold_type = 'youden')
    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1:', f1)
    print('Best threshold:', best_threshold)
    print('specificity:', specificity)
    print('fpr:',fpr)
    print('tpr:',tpr)
    # Plot the ROC curve
    plot_roc_curve(fpr,tpr, roc_auc)
#evaluate_model(X_test,y_test,'fracture_detection_model_2.pt')

