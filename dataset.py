import os

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import wfdb
from scipy.signal import resample
from scipy.signal import savgol_filter

def scaling(X, sigma=0.1):
    scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1, X.shape[1]))
    myNoise = np.matmul(np.ones((X.shape[0], 1)), scalingFactor)
    return X * myNoise

def shift(sig, interval=20):
    for col in range(sig.shape[1]):
        offset = np.random.choice(range(-interval, interval))
        sig[:, col] += offset / 1000
    return sig

def transform(sig, train=False):
    if train:
        if np.random.randn() > 0.5: sig = scaling(sig)
        if np.random.randn() > 0.5: sig = shift(sig)
    return sig

def sg_denoise(ecg_data, window_length, polyorder):
    return savgol_filter(ecg_data, window_length=window_length, polyorder=polyorder, axis=0)

class ECGDataset(Dataset):
    def __init__(self, phase, data_dir, label_csv, folds, leads):
        super(ECGDataset, self).__init__()
        self.phase = phase
        df = pd.read_csv(label_csv)
        df = df[df['fold'].isin(folds)]
        self.data_dir = data_dir
        self.leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        if leads == 'all':
            self.use_leads = np.where(np.in1d(self.leads, self.leads))[0]
        else:
            self.use_leads = np.where(np.in1d(self.leads, leads))[0]
        self.nleads = len(self.use_leads)
        self.labels = df

        self.label_col = 'label'
        self.label_dict = {}

    def __getitem__(self, index: int):
        row = self.labels.iloc[index]
        patient_id = row['id']
        ecg_data, _ = wfdb.rdsamp(os.path.join(self.data_dir, patient_id))
        ecg_data = transform(ecg_data, self.phase == 'train')

        if patient_id.lower().startswith('p'):
            ecg_data = ecg_data[-3500:, self.use_leads]  # shape: (3500, nleads)
            ecg_data = resample(ecg_data, 2800, axis=0)  # shape: (2800, nleads)
            nsteps = ecg_data.shape[0]
        else:
            ecg_data = ecg_data[-2800:, self.use_leads]
            nsteps = ecg_data.shape[0]

        result = np.zeros((2800, self.nleads))  # 7s，400Hz
        ecg_data = sg_denoise(ecg_data, 13, 2)

        result[-nsteps:, :] = ecg_data

        if patient_id in self.label_dict:
            label = self.label_dict[patient_id]
        else:
            label = float(row[self.label_col])
            self.label_dict[patient_id] = label

        return torch.from_numpy(result.transpose()).float(), torch.tensor(label).float(), patient_id

    def __len__(self):
        return len(self.labels)
