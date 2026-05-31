import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
from data_provider.uea import subsample, interpolate_missing, Normalizer
from sktime.datasets import load_from_tsfile_to_dataframe
import warnings

warnings.filterwarnings('ignore')

class RobustTimeSeriesDatasetForCSV_vlbm(Dataset):
    def __init__(self, root_path, flag='train', size=None, scale=True, data_path=None, label_path=None,
                 train_ratio=0.7, val_ratio=0.2, use_anomaly_as_feature=False, interval=5):
        assert flag in ['train', 'val', 'test']
        self.flag = flag
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.scale = scale
        self.root_path = root_path
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.use_anomaly_as_feature = use_anomaly_as_feature
        self.interval = interval
        self.scaler = StandardScaler()
        self.__read_data__()

    def __read_data__(self):
        if self.flag in ['train', 'val']:
            train_data_path = os.path.join(self.root_path, 'train.csv')
            data_df = pd.read_csv(train_data_path)
            self.data = data_df.iloc[:, 1:].values
                
            self.data = np.nan_to_num(self.data)
            self.num_nodes = self.data.shape[1]
            self.anomaly_labels = np.zeros((len(self.data), self.num_nodes))
            
            if self.flag == 'train':
                self.data_x = self.data
                self.anomaly_labels = self.anomaly_labels
            else:  # val
                self.data_x = self.data
                self.anomaly_labels = self.anomaly_labels
                
        else:  # test
            test_data_path = os.path.join(self.root_path, 'test.csv')
            test_label_path = os.path.join(self.root_path, 'test_label.csv')
            
            test_data_df = pd.read_csv(test_data_path)
            self.data = test_data_df.iloc[:, 1:].values
            self.data = np.nan_to_num(self.data)
            self.num_nodes = self.data.shape[1]
            
            test_label_df = pd.read_csv(test_label_path)
            test_labels = test_label_df.iloc[:, 1:].values
            test_labels = np.nan_to_num(test_labels)
            
            if test_labels.ndim == 1:
                test_labels = test_labels.reshape(-1, 1)
            if test_labels.shape[1] != self.num_nodes:
                test_labels = np.repeat(test_labels, self.num_nodes, axis=1)
            
            self.anomaly_labels = test_labels
        
        if self.scale:
            if self.flag == 'train':
                self.scaler.fit(self.data)
                self.data = self.scaler.transform(self.data)
            else:
                train_data_path = os.path.join(self.root_path, 'train.csv')
                train_data_df = pd.read_csv(train_data_path)
                train_data = train_data_df.iloc[:, 1:].values
                train_data = np.nan_to_num(train_data)
                self.scaler.fit(train_data)
                self.data = self.scaler.transform(self.data)
        
        steps_per_day = int(24 * 60 / self.interval)
        total_len = len(self.data)
        time_in_day = np.arange(total_len) % steps_per_day
        day_in_week = (np.arange(total_len) // steps_per_day) % 7
        
        self.data_tid = time_in_day
        self.data_diw = day_in_week
        
        print(f"{self.flag} data shape: {self.data.shape}")
        print(f"{self.flag} labels shape: {self.anomaly_labels.shape}")

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        anomaly_y = self.anomaly_labels[r_begin:r_end]

        seq_x_values = self.data[s_begin:s_end][..., np.newaxis]  # Shape: [seq_len, num_nodes, 1]
        seq_x_tid = self.data_tid[s_begin:s_end]                  # Shape: [seq_len]
        seq_x_diw = self.data_diw[s_begin:s_end]                  # Shape: [seq_len]
        
        seq_x_tid_b = np.tile(seq_x_tid[..., np.newaxis], (1, self.num_nodes))[:, :, np.newaxis]  # [seq_len, num_nodes, 1]
        seq_x_diw_b = np.tile(seq_x_diw[..., np.newaxis], (1, self.num_nodes))[:, :, np.newaxis]  # [seq_len, num_nodes, 1]
        
        seq_x = np.concatenate([seq_x_values, seq_x_tid_b, seq_x_diw_b], axis=-1)  # [seq_len, num_nodes, 3]

        seq_y_values = self.data[r_begin:r_end][..., np.newaxis]  # Shape: [T_out, num_nodes, 1]
        seq_y_tid = self.data_tid[r_begin:r_end]                  # Shape: [T_out]
        seq_y_diw = self.data_diw[r_begin:r_end]                  # Shape: [T_out]

        seq_y_tid_b = np.tile(seq_y_tid[..., np.newaxis], (1, self.num_nodes))[:, :, np.newaxis]  # [T_out, num_nodes, 1]
        seq_y_diw_b = np.tile(seq_y_diw[..., np.newaxis], (1, self.num_nodes))[:, :, np.newaxis]  # [T_out, num_nodes, 1]
        
        seq_y = np.concatenate([seq_y_values, seq_y_tid_b, seq_y_diw_b], axis=-1)  # [T_out, num_nodes, 3]

        seq_x = torch.FloatTensor(seq_x)
        seq_y = torch.FloatTensor(seq_y)
        anomaly_y = torch.FloatTensor(anomaly_y)

        return seq_x, seq_y, anomaly_y

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()

        if data.ndim == 4:  # [Batch, T, N, C]
            values_only = data[..., 0]  
        elif data.ndim == 3:  # [Batch, T, N]
            values_only = data
        else:
            raise ValueError(f"Unsupported data shape: {data.shape}")
        
        batch_size, seq_len, num_nodes = values_only.shape
        data_reshaped = values_only.reshape(-1, num_nodes)
        inversed_data = self.scaler.inverse_transform(data_reshaped)
        
        return inversed_data.reshape(batch_size, seq_len, num_nodes)

class RobustTimeSeriesDatasetForNPY_vlbm_allow(Dataset):
    def __init__(self, root_path, flag='train', size=None, scale=True, data_path=None, label_path=None,
                 train_ratio=0.7, val_ratio=0.2, use_anomaly_as_feature=False, interval=5):
        assert flag in ['train', 'val', 'test']
        self.flag = flag
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.scale = scale
        self.root_path = root_path
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.use_anomaly_as_feature = use_anomaly_as_feature
        self.interval = interval
        self.scaler = StandardScaler()
        
        self.__read_data__()

    def __read_data__(self):
        if self.flag in ['train', 'val']:
            train_data_path = os.path.join(self.root_path, 'train.npy')
            data_df = np.load(train_data_path,allow_pickle=True)

            self.data = data_df
            self.data = np.nan_to_num(self.data)
            self.num_nodes = self.data.shape[1]
            
            self.anomaly_labels = np.zeros((len(self.data), self.num_nodes))
                
        else:  # test
            test_data_path = os.path.join(self.root_path, 'test.npy')
            test_label_path = os.path.join(self.root_path, 'test_label.npy')
            
            test_data_df = np.load(test_data_path,allow_pickle=True)
            self.data = test_data_df
            self.data = np.nan_to_num(self.data)
            self.num_nodes = self.data.shape[1]
            
            test_label_df = np.load(test_label_path,allow_pickle=True).astype("int")
            test_labels = test_label_df
            test_labels = np.nan_to_num(test_labels)
            
            if test_labels.ndim == 1:
                test_labels = test_labels.reshape(-1, 1)
            if test_labels.shape[1] != self.num_nodes:
                test_labels = np.repeat(test_labels, self.num_nodes, axis=1)
            
            self.anomaly_labels = test_labels
        
        if self.scale:
            if self.flag == 'train':
                self.scaler.fit(self.data)
                self.data = self.scaler.transform(self.data)
            else:
                train_data_path = os.path.join(self.root_path, 'train.npy')
                train_data_df = np.load(train_data_path)
                train_data = train_data_df
                train_data = np.nan_to_num(train_data)
                self.scaler.fit(train_data)
                self.data = self.scaler.transform(self.data)
        
        steps_per_day = int(24 * 60 / self.interval)
        total_len = len(self.data)
        time_in_day = np.arange(total_len) % steps_per_day
        day_in_week = (np.arange(total_len) // steps_per_day) % 7
        
        self.data_tid = time_in_day
        self.data_diw = day_in_week
        
        print(f"{self.flag} data shape: {self.data.shape}")
        print(f"{self.flag} labels shape: {self.anomaly_labels.shape}")

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        
        anomaly_y = self.anomaly_labels[r_begin:r_end]

        seq_x_values = self.data[s_begin:s_end][..., np.newaxis]  # Shape: [seq_len, num_nodes, 1]
        seq_x_tid = self.data_tid[s_begin:s_end]                  # Shape: [seq_len]
        seq_x_diw = self.data_diw[s_begin:s_end]                  # Shape: [seq_len]
        
        seq_x_tid_b = np.tile(seq_x_tid[..., np.newaxis], (1, self.num_nodes))[:, :, np.newaxis]  # [seq_len, num_nodes, 1]
        seq_x_diw_b = np.tile(seq_x_diw[..., np.newaxis], (1, self.num_nodes))[:, :, np.newaxis]  # [seq_len, num_nodes, 1]
        
        seq_x = np.concatenate([seq_x_values, seq_x_tid_b, seq_x_diw_b], axis=-1)  # [seq_len, num_nodes, 3]

        seq_y_values = self.data[r_begin:r_end][..., np.newaxis]  # Shape: [T_out, num_nodes, 1]
        seq_y_tid = self.data_tid[r_begin:r_end]                  # Shape: [T_out]
        seq_y_diw = self.data_diw[r_begin:r_end]                  # Shape: [T_out]

        seq_y_tid_b = np.tile(seq_y_tid[..., np.newaxis], (1, self.num_nodes))[:, :, np.newaxis]  # [T_out, num_nodes, 1]
        seq_y_diw_b = np.tile(seq_y_diw[..., np.newaxis], (1, self.num_nodes))[:, :, np.newaxis]  # [T_out, num_nodes, 1]
        
        seq_y = np.concatenate([seq_y_values, seq_y_tid_b, seq_y_diw_b], axis=-1)  # [T_out, num_nodes, 3]

        seq_x = torch.FloatTensor(seq_x)
        seq_y = torch.FloatTensor(seq_y)
        anomaly_y = torch.FloatTensor(anomaly_y)

        return seq_x, seq_y, anomaly_y

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        
        if data.ndim == 4:  # [Batch, T, N, C]
            values_only = data[..., 0] 
        elif data.ndim == 3:  # [Batch, T, N]
            values_only = data
        else:
            raise ValueError(f"Unsupported data shape: {data.shape}")
        
        batch_size, seq_len, num_nodes = values_only.shape
        data_reshaped = values_only.reshape(-1, num_nodes)
        inversed_data = self.scaler.inverse_transform(data_reshaped)
        
        return inversed_data.reshape(batch_size, seq_len, num_nodes)

class RobustTimeSeriesDatasetForNPY_vlbm(Dataset):
    def __init__(self, root_path, flag='train', size=None, scale=True, data_path=None, label_path=None,
                 train_ratio=0.7, val_ratio=0.2, use_anomaly_as_feature=False, interval=5):
        
        assert flag in ['train', 'val', 'test']
        self.flag = flag
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.scale = scale
        self.root_path = root_path
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.use_anomaly_as_feature = use_anomaly_as_feature
        self.interval = interval
        self.scaler = StandardScaler()

        self.__read_data__()

    def __read_data__(self):
        if self.flag in ['train', 'val']:
            train_data_path = os.path.join(self.root_path, 'train.npy')
            data_df = np.load(train_data_path)
            
            self.data = data_df
            self.data = np.nan_to_num(self.data)
            self.num_nodes = self.data.shape[1]
            
            self.anomaly_labels = np.zeros((len(self.data), self.num_nodes))
                
        else:  # test
            test_data_path = os.path.join(self.root_path, 'test.npy')
            test_label_path = os.path.join(self.root_path, 'test_label.npy')
            
            test_data_df = np.load(test_data_path)
            self.data = test_data_df
            self.data = np.nan_to_num(self.data)
            self.num_nodes = self.data.shape[1]
            
            test_label_df = np.load(test_label_path)
            test_labels = test_label_df
            test_labels = np.nan_to_num(test_labels)
            
            if test_labels.ndim == 1:
                test_labels = test_labels.reshape(-1, 1)
            if test_labels.shape[1] != self.num_nodes:
                test_labels = np.repeat(test_labels, self.num_nodes, axis=1)
            
            self.anomaly_labels = test_labels
        
        if self.scale:
            if self.flag == 'train':
                self.scaler.fit(self.data)
                self.data = self.scaler.transform(self.data)
            else:
                train_data_path = os.path.join(self.root_path, 'train.npy')
                train_data_df = np.load(train_data_path)
                train_data = train_data_df
                train_data = np.nan_to_num(train_data)
                self.scaler.fit(train_data)
                self.data = self.scaler.transform(self.data)
        
        steps_per_day = int(24 * 60 / self.interval)
        total_len = len(self.data)
        time_in_day = np.arange(total_len) % steps_per_day
        day_in_week = (np.arange(total_len) // steps_per_day) % 7
        
        self.data_tid = time_in_day
        self.data_diw = day_in_week
        
        print(f"{self.flag} data shape: {self.data.shape}")
        print(f"{self.flag} labels shape: {self.anomaly_labels.shape}")

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        anomaly_y = self.anomaly_labels[r_begin:r_end]

        seq_x_values = self.data[s_begin:s_end][..., np.newaxis]  # Shape: [seq_len, num_nodes, 1]
        seq_x_tid = self.data_tid[s_begin:s_end]                  # Shape: [seq_len]
        seq_x_diw = self.data_diw[s_begin:s_end]                  # Shape: [seq_len]
        
        seq_x_tid_b = np.tile(seq_x_tid[..., np.newaxis], (1, self.num_nodes))[:, :, np.newaxis]  # [seq_len, num_nodes, 1]
        seq_x_diw_b = np.tile(seq_x_diw[..., np.newaxis], (1, self.num_nodes))[:, :, np.newaxis]  # [seq_len, num_nodes, 1]
        
        seq_x = np.concatenate([seq_x_values, seq_x_tid_b, seq_x_diw_b], axis=-1)  # [seq_len, num_nodes, 3]

        seq_y_values = self.data[r_begin:r_end][..., np.newaxis]  # Shape: [T_out, num_nodes, 1]
        seq_y_tid = self.data_tid[r_begin:r_end]                  # Shape: [T_out]
        seq_y_diw = self.data_diw[r_begin:r_end]                  # Shape: [T_out]

        seq_y_tid_b = np.tile(seq_y_tid[..., np.newaxis], (1, self.num_nodes))[:, :, np.newaxis]  # [T_out, num_nodes, 1]
        seq_y_diw_b = np.tile(seq_y_diw[..., np.newaxis], (1, self.num_nodes))[:, :, np.newaxis]  # [T_out, num_nodes, 1]
        
        seq_y = np.concatenate([seq_y_values, seq_y_tid_b, seq_y_diw_b], axis=-1)  # [T_out, num_nodes, 3]

        seq_x = torch.FloatTensor(seq_x)
        seq_y = torch.FloatTensor(seq_y)
        anomaly_y = torch.FloatTensor(anomaly_y)

        return seq_x, seq_y, anomaly_y

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        
        if data.ndim == 4:  # [Batch, T, N, C]
            values_only = data[..., 0]  
        elif data.ndim == 3:  # [Batch, T, N]
            values_only = data
        else:
            raise ValueError(f"Unsupported data shape: {data.shape}")
        
        batch_size, seq_len, num_nodes = values_only.shape
        data_reshaped = values_only.reshape(-1, num_nodes)
        inversed_data = self.scaler.inverse_transform(data_reshaped)
        
        return inversed_data.reshape(batch_size, seq_len, num_nodes)

class RobustTimeSeriesDatasetForNPY0_vlbm(Dataset):
    def __init__(self, root_path, data_path='data_merged.npz', label_path='anomaly_labels.npy',flag='train',
                 size=None, scale=True,
                 train_ratio=0.6, val_ratio=0.2,interval=None):
        assert flag in ['train', 'val', 'test']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.scale = scale
        self.root_path = root_path
        self.data_filename = data_path
        self.label_filename = label_path
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.interval = interval

        self.__read_data__()

    def __read_data__(self):
        file_datapath = os.path.join(self.root_path, self.data_filename)
        label_datapath = os.path.join(self.root_path, self.label_filename)
        label = np.load(label_datapath)
        data = np.load(file_datapath)[:,:,0]
        self.num_nodes = data.shape[1]

        num_train = int(len(data) * self.train_ratio)
        num_val = int(len(data) * self.val_ratio)
        num_test = len(data) - num_train - num_val
        
        border1s = [0, num_train - self.seq_len, len(data) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_val, len(data)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.scale:
            self.scaler = StandardScaler()
            train_data_for_fit = data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data_for_fit)
            data = self.scaler.transform(data)

        self.data = data[border1:border2]
        self.anomaly_labels = label[border1:border2]
        steps_per_day = int(24 * 60 / self.interval)
        total_len = len(data)
        time_in_day = np.arange(total_len) % steps_per_day
        day_in_week = (np.arange(total_len) // steps_per_day) % 7
        self.data_tid = time_in_day[border1:border2]
        self.data_diw = day_in_week[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        anomaly_y = self.anomaly_labels[r_begin:r_end]

        seq_x_values = self.data[s_begin:s_end][..., np.newaxis]
        seq_x_tid = self.data_tid[s_begin:s_end]
        seq_x_diw = self.data_diw[s_begin:s_end]
        seq_x_tid_b = np.tile(seq_x_tid[..., np.newaxis], (1, self.num_nodes))[:, :, np.newaxis]
        seq_x_diw_b = np.tile(seq_x_diw[..., np.newaxis], (1, self.num_nodes))[:, :, np.newaxis]
        seq_x = np.concatenate([seq_x_values, seq_x_tid_b, seq_x_diw_b], axis=-1)
            
        seq_y_values = self.data[r_begin:r_end][..., np.newaxis] # Shape: [T_out, N, 1]
        seq_y_tid = self.data_tid[r_begin:r_end]                         # Shape: [T_out]
        seq_y_diw = self.data_diw[r_begin:r_end]                         # Shape: [T_out]

        seq_y_tid_b = np.tile(seq_y_tid[..., np.newaxis], (1, self.num_nodes))[:, :, np.newaxis] # Shape: [T_out, N, 1]
        seq_y_diw_b = np.tile(seq_y_diw[..., np.newaxis], (1, self.num_nodes))[:, :, np.newaxis] # Shape: [T_out, N, 1]
        seq_y = np.concatenate([seq_y_values, seq_y_tid_b, seq_y_diw_b], axis=-1)
        return seq_x, seq_y, anomaly_y

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        values_only = data[..., 0] # Shape: [Batch, T, N]
        num_nodes = values_only.shape[-1] 
        data_reshaped = values_only.reshape(-1, num_nodes)
        inversed_data = self.scaler.inverse_transform(data_reshaped)
        return inversed_data.reshape(values_only.shape)
    
class RobustTimeSeriesDatasetForNPY1_vlbm(Dataset):
    def __init__(self, root_path, data_path='data_merged.npz', label_path='anomaly_labels.npy',flag='train',
                 size=None, scale=True,
                 train_ratio=0.6, val_ratio=0.2,interval=None):
        assert flag in ['train', 'val', 'test']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.scale = scale
        self.root_path = root_path
        self.data_filename = data_path
        self.label_filename = label_path
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.interval = interval

        self.__read_data__()

    def __read_data__(self):
        file_datapath = os.path.join(self.root_path, self.data_filename)
        label_datapath = os.path.join(self.root_path, self.label_filename)
        label = np.load(label_datapath)
        data = np.load(file_datapath)[:,:,1]
        self.num_nodes = data.shape[1]

        num_train = int(len(data) * self.train_ratio)
        num_val = int(len(data) * self.val_ratio)
        num_test = len(data) - num_train - num_val
        
        border1s = [0, num_train - self.seq_len, len(data) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_val, len(data)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.scale:
            self.scaler = StandardScaler()
            train_data_for_fit = data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data_for_fit)
            data = self.scaler.transform(data)

        self.data = data[border1:border2]
        self.anomaly_labels = label[border1:border2]
        steps_per_day = int(24 * 60 / self.interval)
        total_len = len(data)
        time_in_day = np.arange(total_len) % steps_per_day
        day_in_week = (np.arange(total_len) // steps_per_day) % 7
        self.data_tid = time_in_day[border1:border2]
        self.data_diw = day_in_week[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        anomaly_y = self.anomaly_labels[r_begin:r_end]
        seq_x_values = self.data[s_begin:s_end][..., np.newaxis]
        seq_x_tid = self.data_tid[s_begin:s_end]
        seq_x_diw = self.data_diw[s_begin:s_end]
        seq_x_tid_b = np.tile(seq_x_tid[..., np.newaxis], (1, self.num_nodes))[:, :, np.newaxis]
        seq_x_diw_b = np.tile(seq_x_diw[..., np.newaxis], (1, self.num_nodes))[:, :, np.newaxis]
        seq_x = np.concatenate([seq_x_values, seq_x_tid_b, seq_x_diw_b], axis=-1)
            
        seq_y_values = self.data[r_begin:r_end][..., np.newaxis] # Shape: [T_out, N, 1]
        seq_y_tid = self.data_tid[r_begin:r_end]                         # Shape: [T_out]
        seq_y_diw = self.data_diw[r_begin:r_end]                         # Shape: [T_out]

        seq_y_tid_b = np.tile(seq_y_tid[..., np.newaxis], (1, self.num_nodes))[:, :, np.newaxis] # Shape: [T_out, N, 1]
        seq_y_diw_b = np.tile(seq_y_diw[..., np.newaxis], (1, self.num_nodes))[:, :, np.newaxis] # Shape: [T_out, N, 1]
        seq_y = np.concatenate([seq_y_values, seq_y_tid_b, seq_y_diw_b], axis=-1)
        return seq_x, seq_y, anomaly_y

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        values_only = data[..., 1] # Shape: [Batch, T, N]
        num_nodes = values_only.shape[-1] 
        data_reshaped = values_only.reshape(-1, num_nodes)
        inversed_data = self.scaler.inverse_transform(data_reshaped)
        return inversed_data.reshape(values_only.shape)

class Dataset_PEMS_vlbm_ConsistentWithV1(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', interval=None):
        self.features = features
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
            
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        
        self.target = target
        self.scale = scale
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.interval = interval
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        data_file = os.path.join(self.root_path, self.data_path)
        data_np = np.load(data_file, allow_pickle=True)
        data_np = data_np['data'][:, :, 0] # [Total_Len, Num_Nodes]
        
        self.num_nodes = data_np.shape[1]
        _len = len(data_np)

        train_ratio = 0.6
        valid_ratio = 0.2

        border_train_end = int(train_ratio * _len)
        border_val_end = int((train_ratio + valid_ratio) * _len)

        train_data_raw = data_np[:border_train_end]
        # val_data_raw = data_np[border_train_end:border_val_end] # 与第一版逻辑同步
        # test_data_raw = data_np[border_val_end:]               # 与第一版逻辑同步
        
        if self.scale:
            self.scaler.fit(train_data_raw)
            data_scaled = self.scaler.transform(data_np)
        else:
            data_scaled = data_np

        data_list = [
            data_scaled[:border_train_end],                          # Train
            data_scaled[border_train_end:border_val_end],           # Val
            data_scaled[border_val_end:]                            # Test
        ]
        self.data = data_list[self.set_type]

        steps_per_day = int(24 * 60 / self.interval)
        time_in_day = np.arange(_len) % steps_per_day
        day_in_week = (np.arange(_len) // steps_per_day) % 7
        
        tid_list = [
            time_in_day[:border_train_end],
            time_in_day[border_train_end:border_val_end],
            time_in_day[border_val_end:]
        ]
        diw_list = [
            day_in_week[:border_train_end],
            day_in_week[border_train_end:border_val_end],
            day_in_week[border_val_end:]
        ]
        
        self.data_tid = tid_list[self.set_type]
        self.data_diw = diw_list[self.set_type]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x_values = self.data[s_begin:s_end][..., np.newaxis]
        seq_x_tid = self.data_tid[s_begin:s_end]
        seq_x_diw = self.data_diw[s_begin:s_end]
        seq_x_tid_b = np.tile(seq_x_tid[..., np.newaxis], (1, self.num_nodes))[:, :, np.newaxis]
        seq_x_diw_b = np.tile(seq_x_diw[..., np.newaxis], (1, self.num_nodes))[:, :, np.newaxis]
        seq_x = np.concatenate([seq_x_values, seq_x_tid_b, seq_x_diw_b], axis=-1)
        
        seq_y_values = self.data[r_begin:r_end][..., np.newaxis]
        seq_y_tid = self.data_tid[r_begin:r_end]
        seq_y_diw = self.data_diw[r_begin:r_end]
        seq_y_tid_b = np.tile(seq_y_tid[..., np.newaxis], (1, self.num_nodes))[:, :, np.newaxis]
        seq_y_diw_b = np.tile(seq_y_diw[..., np.newaxis], (1, self.num_nodes))[:, :, np.newaxis]
        seq_y = np.concatenate([seq_y_values, seq_y_tid_b, seq_y_diw_b], axis=-1)
        
        return seq_x, seq_y

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        if data.shape[-1] == 3:
            data = data[..., 0]
        return self.scaler.inverse_transform(data)

class Dataset_Custom_vlbm(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',interval=None):
        self.features = features
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.target = target
        self.scale = scale
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.interval = interval
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        cols = list(df_raw.columns)
        if 'date' in cols:
            cols.remove('date')
        if self.target in cols:
            cols.remove(self.target)
            cols.append(self.target)
        df_raw = df_raw[['date'] + cols]
        self.num_nodes = len(cols)
        df_values = df_raw[cols]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        train_values = df_values.iloc[border1s[0]:border2s[0]].values
        if self.scale:
            self.scaler.fit(train_values)
            all_values_scaled = self.scaler.transform(df_values.values)
        else:
            all_values_scaled = df_values.values
        steps_per_day = int(24 * 60 / self.interval)
        total_len = len(df_raw)
        time_in_day = np.arange(total_len) % steps_per_day
        day_in_week = (np.arange(total_len) // steps_per_day) % 7
        self.data_values = all_values_scaled[border1:border2]
        self.data_tid = time_in_day[border1:border2]
        self.data_diw = day_in_week[border1:border2]


    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x_values = self.data_values[s_begin:s_end][..., np.newaxis]
        seq_x_tid = self.data_tid[s_begin:s_end]
        seq_x_diw = self.data_diw[s_begin:s_end]
        seq_x_tid_b = np.tile(seq_x_tid[..., np.newaxis], (1, self.num_nodes))[:, :, np.newaxis]
        seq_x_diw_b = np.tile(seq_x_diw[..., np.newaxis], (1, self.num_nodes))[:, :, np.newaxis]
        seq_x = np.concatenate([seq_x_values, seq_x_tid_b, seq_x_diw_b], axis=-1)

        seq_y_values = self.data_values[r_begin:r_end][..., np.newaxis] # Shape: [T_out, N, 1]
        seq_y_tid = self.data_tid[r_begin:r_end]                         # Shape: [T_out]
        seq_y_diw = self.data_diw[r_begin:r_end]                         # Shape: [T_out]

        seq_y_tid_b = np.tile(seq_y_tid[..., np.newaxis], (1, self.num_nodes))[:, :, np.newaxis] # Shape: [T_out, N, 1]
        seq_y_diw_b = np.tile(seq_y_diw[..., np.newaxis], (1, self.num_nodes))[:, :, np.newaxis] # Shape: [T_out, N, 1]
        seq_y = np.concatenate([seq_y_values, seq_y_tid_b, seq_y_diw_b], axis=-1)
        return seq_x, seq_y

    def __len__(self):
        return len(self.data_values) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        num_nodes = values_only.shape[-1] 
        
        data_reshaped = values_only.reshape(-1, num_nodes)
        
        inversed_data = self.scaler.inverse_transform(data_reshaped)
        
        return inversed_data.reshape(values_only.shape)


    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', cycle=None):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.cycle = cycle

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        data_file = os.path.join(self.root_path, self.data_path)
        data = np.load(data_file, allow_pickle=True)
        data = data['data'][:, :, 0]

        num_train = int(len(data) * 0.6)
        num_test = int(len(data) * 0.2)
        num_valid = int(len(data) * 0.2)
        border1s = [0, num_train - self.seq_len, len(data) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_valid, len(data)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.scale:
            train_data = data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        # add cycle
        self.cycle_index = (np.arange(len(data)) % self.cycle)[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))

        cycle_index = torch.tensor(self.cycle_index[s_end])

        return seq_x, seq_y, seq_x_mark, seq_y_mark, cycle_index

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
