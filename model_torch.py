import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, KFold
import timm
import random
from data_torch import apply_augmentation, apply_augmentation_multi

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_model_multichannel(num_channels):
    #handles any number of channels for inputs, depending on the shape of X_data
    base_model = timm.create_model("tf_efficientnetv2_s", pretrained=True, in_chans=num_channels)  # Update the number of input channels
    num_features = base_model.classifier.in_features
    base_model.classifier = nn.Linear(num_features, 1)
    base_model.classifier.out_features = 1
    base_model.classifier.bias.data.fill_(0.0)
    base_model.classifier.weight.data.normal_(0.0, 0.02)
    base_model.act = nn.Sigmoid()
    
    optimizer = Adam(base_model.parameters(), lr=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    return base_model, optimizer, loss_fn

def ensemble_predict(models, X):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.tensor(X, dtype=torch.float32).to(device)

    predictions = []
    with torch.no_grad():
        for model in models:
            model = model.to(device)
            pred = model(X)
            predictions.append(pred.cpu().numpy())
        
    ensemble_prediction = np.mean(predictions, axis=0)
    return ensemble_prediction

def single_predict(model, X):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.tensor(X, dtype=torch.float32).to(device)

    with torch.no_grad():
        model = model.to(device)
        pred = model(X)
    return pred.cpu().numpy()

def train_model_multi(model, optimizer, loss_fn, X, y, cross_val, epochs=12, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    for train_index, val_index in cross_val.split(X, y):
        X_train_fold, X_val_fold = X[train_index], X[val_index]
        y_train_fold, y_val_fold = y[train_index], y[val_index]

        # Create DataLoader for training and validation sets
        train_dataset = CustomDataset(X_train_fold, y_train_fold)
        val_dataset = CustomDataset(X_val_fold, y_val_fold)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        for epoch in range(epochs):
            model.train()
            
            # Train on mini-batches
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = loss_fn(outputs, batch_y)
                loss.backward()
                optimizer.step()

            # Evaluate on validation set
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch_val_X, batch_val_y in val_loader:
                    batch_val_X, batch_val_y = batch_val_X.to(device), batch_val_y.to(device)
                    val_outputs = model(batch_val_X)
                    val_loss = loss_fn(val_outputs, batch_val_y)
                    val_losses.append(val_loss.item())

            # Calculate average validation loss
            avg_val_loss = sum(val_losses) / len(val_losses)

            print(f'Epoch: {epoch + 1}, Loss: {loss.item()}, Val Loss: {avg_val_loss}')

    return model

def train_model_single_validation(model, optimizer, loss_fn, X_train, y_train, X_val, y_val, epochs=12, batch_size=32):
    # like train model multi but without cross validation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Create DataLoader for training and validation sets
    train_dataset = CustomDataset(X_train, y_train)
    val_dataset = CustomDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(epochs):
        model.train()
        
        # Train on mini-batches
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            optimizer.step()

        # Evaluate on validation set
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_val_X, batch_val_y in val_loader:
                batch_val_X, batch_val_y = batch_val_X.to(device), batch_val_y.to(device)
                val_outputs = model(batch_val_X)
                val_loss = loss_fn(val_outputs, batch_val_y)
                val_losses.append(val_loss.item())

        # Calculate average validation loss
        avg_val_loss = sum(val_losses) / len(val_losses)

        print(f'Epoch: {epoch + 1}, Loss: {loss.item()}, Val Loss: {avg_val_loss}')

    return model


### No longer used ###

def create_model():
    base_model = timm.create_model("tf_efficientnetv2_s", pretrained=True, in_chans=1)
    num_features = base_model.classifier.in_features
    base_model.classifier = nn.Linear(num_features, 1)
    base_model.classifier.out_features = 1
    base_model.classifier.bias.data.fill_(0.0)
    base_model.classifier.weight.data.normal_(0.0, 0.02)
    base_model.act = nn.Sigmoid()
    
    optimizer = Adam(base_model.parameters(), lr=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    return base_model, optimizer, loss_fn

def create_2_model():
    #deals with 2 channels, sag and coronal images
    base_model = timm.create_model("tf_efficientnetv2_s", pretrained=True, in_chans=2)  # Update the number of input channels
    num_features = base_model.classifier.in_features
    base_model.classifier = nn.Linear(num_features, 1)
    base_model.classifier.out_features = 1
    base_model.classifier.bias.data.fill_(0.0)
    base_model.classifier.weight.data.normal_(0.0, 0.02)
    base_model.act = nn.Sigmoid()
    
    optimizer = Adam(base_model.parameters(), lr=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    return base_model, optimizer, loss_fn

def train_model(model, optimizer, loss_fn, X, y, cross_val,apply_data_augmentation = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    X = torch.tensor(X, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.float32).to(device)

    for train_index, val_index in cross_val.split(X, y):
        X_train_fold, X_val_fold = X[train_index], X[val_index]
        y_train_fold, y_val_fold = y[train_index], y[val_index]
        # Apply data augmentation only to the training part of the fold
        if apply_data_augmentation:
            X_train, y_train = apply_augmentation(X_train_fold,y_train_fold)

        for epoch in range(10):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_fold)
            loss = loss_fn(outputs, y_train_fold)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_fold)
                val_loss = loss_fn(val_outputs, y_val_fold)
                print(f'Epoch: {epoch + 1}, Loss: {loss.item()}, Val Loss: {val_loss.item()}')

    return model
