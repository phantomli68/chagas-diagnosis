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

        self.leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                      'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        self.use_leads = np.where(np.in1d(self.leads, self.leads))[0]
        self.nleads = len(self.use_leads)

        self.patient_ids = []

        # ✅ 递归查找所有 .hea 文件并筛选符合要求的数据
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.hea'):
                    hea_path = os.path.join(root, file)
                    source = None
                    with open(hea_path, 'r') as f:
                        for line in f:
                            if '# Source:' in line:
                                source = line.strip().split(':')[-1].strip()
                                break
                    if source in ['SaMi-Trop', 'PTB-XL']:
                        patient_id = file[:-4]
                        self.patient_ids.append((root, patient_id))

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, index):
        folder_path, patient_id = self.patient_ids[index]
        hea_path = os.path.join(folder_path, patient_id + '.hea')

        # 读取标签 & 数据来源
        label = 0
        source = None
        with open(hea_path, 'r') as f:
            for line in f:
                if 'Chagas label:' in line:
                    label = 1 if 'True' in line else 0
                elif '# Source:' in line:
                    source = line.strip().split(':')[-1].strip()

        # 读取 ECG 信号
        record_path = os.path.join(folder_path, patient_id)
        ecg_data, _ = wfdb.rdsamp(record_path)

        # ✅ 根据 source 判断处理方式
        if source == 'PTB-XL':
            ecg_data = ecg_data[-3500:, self.use_leads]
            ecg_data = resample(ecg_data, 2800, axis=0)
            nsteps = ecg_data.shape[0]
        elif source == 'SaMi-Trop':
            ecg_data = ecg_data[-2800:, self.use_leads]
            nsteps = ecg_data.shape[0]


        result = np.zeros((2800, self.nleads))
        ecg_data = sg_denoise(ecg_data, 13, 2)
        result[-nsteps:, :] = ecg_data

        return torch.from_numpy(result.transpose()).float(), torch.tensor(label).float(), patient_id
