#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import joblib
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import sys

from helper_code import *

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments for the functions.
#
################################################################################

# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.
import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import random_split, DataLoader
from ecg_dataset import ECGDataset
from vHeat import vHeat1D
from loss import FocalLoss
from utils import cal_f1s, split_data
from scipy.signal import savgol_filter
from scipy.signal import resample

# Train your model.
def train_model(data_folder, model_folder, verbose):
    batch_size = 64
    num_workers = 16
    epochs = 80
    learning_rate = 1e-4
    use_gpu = torch.cuda.is_available()
    best_f1 = 0

    device = torch.device('cuda:0' if use_gpu else 'cpu')
    if verbose:
        print(f"Using device: {device}")

    # check_all_sources(data_folder)

    # The data_folder contains all the processed .hea and .dat files from the PTB-XL and SaMi-Trop datasets.

    dataset = ECGDataset(data_folder)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)

    model = vHeat1D().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = FocalLoss(alpha=0.85, gamma=2.0, reduction="mean").to(device)



    print("Training model...")
    print(f"  - epochs: {epochs}")
    # 训练过程
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        outputs_list, labels_list = [], []

        for data, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", disable=not verbose):
            data, labels = data.to(device), labels.to(device).view(-1, 1).float()
            preds = model(data)
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            outputs_list.append(preds.detach().cpu().numpy())
            labels_list.append(labels.detach().cpu().numpy())

        scheduler.step()

        if verbose:
            print(f"[Epoch {epoch+1}] Training loss: {epoch_loss:.4f}")

        # 验证
        model.eval()
        val_outputs, val_labels = [], []
        val_loss = 0

        with torch.no_grad():
            for data, labels, _ in val_loader:
                data, labels = data.to(device), labels.to(device).view(-1, 1).float()
                preds = model(data)
                loss = criterion(preds, labels)
                preds = torch.sigmoid(preds)

                val_outputs.append(preds.cpu().numpy())
                val_labels.append(labels.cpu().numpy())
                val_loss += loss.item()

        y_true = np.vstack(val_labels)
        y_score = np.vstack(val_outputs)
        f1_scores = cal_f1s(y_true, y_score)
        avg_f1 = np.mean(f1_scores)

        if verbose:
            print(f"[Epoch {epoch+1}] Validation loss: {val_loss:.4f}, F1: {f1_scores}, Avg F1: {avg_f1:.4f}")

        # 保存模型
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            os.makedirs(model_folder, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(model_folder, 'model.pth'))
            if verbose:
                print(f"Saved best model with Avg F1: {avg_f1:.4f}")

    if verbose:
        print("Training complete.")

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def load_model(model_folder, verbose=False):
    model_path = os.path.join(model_folder, "model.pth")
    model = vHeat1D()
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
    model.eval()
    return {"model": model}

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_model(record, model, verbose=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = model["model"]
    net.eval()
    net.to(device)

    ecg_data, fields = load_signals(record)

    if fields['fs'] != 400:
        ecg_data = ecg_data[-3500:, :]  # 取所有12导联
        ecg_data = resample(ecg_data, 2800, axis=0)
        nsteps = ecg_data.shape[0]
    else:
        ecg_data = ecg_data[-2800:, :]
        nsteps = ecg_data.shape[0]

    ecg_data = sg_denoise(ecg_data, 13, 2)

    data = np.zeros((2800, 12))
    data[-nsteps:, :] = ecg_data

    data = torch.tensor(data, dtype=torch.float32)
    data = data.unsqueeze(0)  # batch size 1
    data = data.permute(0, 2, 1)  # 调整为 (batch, channels, length)
    data = data.to(device)

    with torch.no_grad():
        output = net(data)  # shape: [1, 1]
        prob = torch.sigmoid(output).item()

    binary_output = int(prob >= 0.56)

    if verbose:
        print(f"Predicted probability: {prob:.6f}, binary: {binary_output}")

    return binary_output, prob

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Extract your features.
def extract_source(record):
    header = load_header(record)
    # Extract the source from the record (but do not use it as a feature).
    source = get_source(header)
    return source

# Save your trained model.
def save_model(model_folder, model):
    d = {'model': model}
    filename = os.path.join(model_folder, 'model.sav')
    joblib.dump(d, filename, protocol=0)


def sg_denoise(ecg_data, window_length, polyorder):
    return savgol_filter(ecg_data, window_length=window_length, polyorder=polyorder, axis=0)

# def check_all_sources(data_folder):
#     invalid_records = []
#     for filename in os.listdir(data_folder):
#         if filename.endswith('.hea'):
#             record_path = os.path.join(data_folder, filename[:-4])  # remove .hea extension
#             source = extract_source(record_path)
#             if source not in ["PTB-XL", "SaMi-Trop"]:
#                 invalid_records.append((record_path, source))
# 
#     if invalid_records:
#         print("The following records are not from PTB-XL or SaMi-Trop:")
#         for path, src in invalid_records:
#             print(f"  - {path} -> {src}")
#     else:
#         print("✅ All records are from PTB-XL or SaMi-Trop.")
