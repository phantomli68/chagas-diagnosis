import os
import torch
import numpy as np
import wfdb
from torch.utils.data import Dataset
from scipy.signal import savgol_filter
from scipy.signal import resample

def sg_denoise(ecg, window=13, poly=2):
    return savgol_filter(ecg, window_length=window, polyorder=poly, axis=0)

class ECGDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        super(ECGDataset, self).__init__()
        self.data_dir = data_dir
        self.transform = transform

        self.leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        self.use_leads = np.where(np.in1d(self.leads, self.leads))[0]
        self.nleads = len(self.use_leads)

        target_folders = ["PTB-XL", "SaMi-Trop"]
        self.patient_ids = []

        for folder in target_folders:
            folder_path = os.path.join(data_dir, folder)
            if os.path.isdir(folder_path):
                # ✅ 递归查找所有 .hea 文件
                for root, _, files in os.walk(folder_path):
                    for file in files:
                        if file.endswith('.hea'):
                            patient_id = file[:-4]  # 去掉 .hea
                            self.patient_ids.append((root, patient_id))

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, index):
        folder_path, patient_id = self.patient_ids[index]
        hea_path = os.path.join(folder_path, patient_id + '.hea')

        # 读取标签
        label = 0
        with open(hea_path, 'r') as f:
            for line in f:
                if 'Chagas label:' in line:
                    label = 1 if 'True' in line else 0
                    break

        # 读取 ECG 信号
        record_path = os.path.join(folder_path, patient_id)
        ecg_data, _ = wfdb.rdsamp(record_path)

        # 数据截取与重采样
        if patient_id.lower().startswith('p'):
            ecg_data = ecg_data[-3500:, self.use_leads]
            ecg_data = resample(ecg_data, 2800, axis=0)
            nsteps = ecg_data.shape[0]
        else:
            ecg_data = ecg_data[-2800:, self.use_leads]
            nsteps = ecg_data.shape[0]

        result = np.zeros((2800, self.nleads))
        ecg_data = sg_denoise(ecg_data, 13, 2)
        result[-nsteps:, :] = ecg_data

        return torch.from_numpy(result.transpose()).float(), torch.tensor(label).float(), patient_idimport os
import torch
import numpy as np
import wfdb
from torch.utils.data import Dataset
from scipy.signal import savgol_filter
from scipy.signal import resample

def sg_denoise(ecg, window=13, poly=2):
    return savgol_filter(ecg, window_length=window, polyorder=poly, axis=0)

class ECGDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        super(ECGDataset, self).__init__()
        self.data_dir = data_dir
        self.transform = transform

        self.leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        self.use_leads = np.where(np.in1d(self.leads, self.leads))[0]
        self.nleads = len(self.use_leads)

        target_folders = ["PTB-XL", "SaMi-Trop"]
        self.patient_ids = []

        for folder in target_folders:
            folder_path = os.path.join(data_dir, folder)
            if os.path.isdir(folder_path):
                # ✅ 递归查找所有 .hea 文件
                for root, _, files in os.walk(folder_path):
                    for file in files:
                        if file.endswith('.hea'):
                            patient_id = file[:-4]  # 去掉 .hea
                            self.patient_ids.append((root, patient_id))

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, index):
        folder_path, patient_id = self.patient_ids[index]
        hea_path = os.path.join(folder_path, patient_id + '.hea')

        # 读取标签
        label = 0
        with open(hea_path, 'r') as f:
            for line in f:
                if 'Chagas label:' in line:
                    label = 1 if 'True' in line else 0
                    break

        # 读取 ECG 信号
        record_path = os.path.join(folder_path, patient_id)
        ecg_data, _ = wfdb.rdsamp(record_path)

        # 数据截取与重采样
        if patient_id.lower().startswith('p'):
            ecg_data = ecg_data[-3500:, self.use_leads]
            ecg_data = resample(ecg_data, 2800, axis=0)
            nsteps = ecg_data.shape[0]
        else:
            ecg_data = ecg_data[-2800:, self.use_leads]
            nsteps = ecg_data.shape[0]

        result = np.zeros((2800, self.nleads))
        ecg_data = sg_denoise(ecg_data, 13, 2)
        result[-nsteps:, :] = ecg_data

        return torch.from_numpy(result.transpose()).float(), torch.tensor(label).float(), patient_id
