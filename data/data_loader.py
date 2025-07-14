"""Building blocks for TimeArtist.

Copyright (2025) TimeArtist's Author

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Reference:
    https://github.com/TimeArtist-AAAI2026
"""


import os
import torch
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from data.timefeatures import time_features

import warnings
warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 seasonal_patterns=None, subset_rand_ratio=1.0):
        self.subset_rand_ratio = subset_rand_ratio
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 0
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = 0
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        if self.set_type == 0:
            self.internal = int(1 // self.subset_rand_ratio)
        else:
            self.internal = 1

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data = data
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        if self.set_type == 0:
            index = index * self.internal
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        if self.set_type == 0:
            return max(int((len(self.data_x) - self.seq_len - self.pred_len + 1) * self.subset_rand_ratio), 1)
        else:
            return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t',
                 seasonal_patterns=None, subset_rand_ratio=1.0):
        self.subset_rand_ratio = subset_rand_ratio
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        if self.set_type == 0:
            self.internal = int(1 // self.subset_rand_ratio)
        else:
            self.internal = 1

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data = data
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        if self.set_type == 0:
            index = index * self.internal
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        if self.set_type == 0:
            return max(int((len(self.data_x) - self.seq_len - self.pred_len + 1) * self.subset_rand_ratio), 1)
        else:
            return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 seasonal_patterns=None, subset_rand_ratio=1.0):
        self.subset_rand_ratio = subset_rand_ratio
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        if self.set_type == 0:
            self.internal = int(1 // self.subset_rand_ratio)
        else:
            self.internal = 1

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data = data
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        if self.set_type == 0:
            index = index * self.internal
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        if self.set_type == 0:
            return max(int((len(self.data_x) - self.seq_len - self.pred_len + 1) * self.subset_rand_ratio), 1)
        else:
            return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


# TimeMixer MultiVariate Short-term forecasting
class Dataset_PEMS(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 seasonal_patterns=None, subset_rand_ratio=1.0):
        self.subset_rand_ratio = subset_rand_ratio
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        if self.set_type == 0:
            self.internal = int(1 // self.subset_rand_ratio)
        else:
            self.internal = 1

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        data_file = os.path.join(self.root_path, self.data_path)
        data = np.load(data_file, allow_pickle=True)
        data = data['data'][:, :, 0]
        self.data = data.copy()

        train_ratio = 0.6
        valid_ratio = 0.2
        train_data = data[:int(train_ratio * len(data))]
        valid_data = data[int(train_ratio * len(data)):int((train_ratio + valid_ratio) * len(data))]
        test_data = data[int((train_ratio + valid_ratio) * len(data)):]
        total_data = [train_data, valid_data, test_data]
        data = total_data[self.set_type]

        if self.scale:
            self.scaler.fit(data)
            data = self.scaler.transform(data)

        df = pd.DataFrame(data)
        df = df.fillna(method='ffill', limit=len(df)).fillna(method='bfill', limit=len(df)).values

        self.data_x = df
        self.data_y = df

    def __getitem__(self, index):
        if self.set_type == 2:  # test:首尾相连
            s_begin = index * 12
        else:
            s_begin = index

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_y.shape[0], 1))

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        if self.set_type == 0:
            return max(int((len(self.data_x) - self.seq_len - self.pred_len + 1) * self.subset_rand_ratio), 1)
        elif self.set_type == 2:
            return max(int(len(self.data_x) - self.seq_len - self.pred_len + 1) // 12, 1)
        else:
            return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


# TimeMixer MultiVariate Short-term forecasting
class Dataset_Solar(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 seasonal_patterns=None, subset_rand_ratio=1.0):
        self.subset_rand_ratio = subset_rand_ratio
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        if self.set_type == 0:
            self.internal = int(1 // self.subset_rand_ratio)
        else:
            self.internal = 1

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = []
        with open(os.path.join(self.root_path, self.data_path), "r", encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n').split(',')  # 去除文本中的换行符
                data_line = np.stack([float(i) for i in line])
                df_raw.append(data_line)
        df_raw = np.stack(df_raw, 0)
        df_raw = pd.DataFrame(df_raw)
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_data = df_raw.values
        self.data = df_data.copy()

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_data)
        else:
            data = df_data

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        if self.set_type == 0:
            index = index * self.internal

        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_y.shape[0], 1))

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        if self.set_type == 0:
            return max(int((len(self.data_x) - self.seq_len - self.pred_len + 1) * self.subset_rand_ratio), 1)
        else:
            return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    

def plot_sequence(true_orig, save_name):
    import numpy as np
    import matplotlib.pyplot as plt

    plt.figure(figsize=(15, 2.5))

    plt.plot(
        np.arange(true_orig.shape[0]),
        true_orig,
        linewidth=1,
    )

    plt.savefig(f'{save_name}.png', bbox_inches='tight', pad_inches=0.05, dpi=1000)
    plt.savefig(f'{save_name}.svg', bbox_inches='tight', pad_inches=0.05, dpi=1000)
    plt.close()


def plot_datasets(dataloader, save_path):
    print(f'########### {save_path}')
    tensor2d = dataloader.data_x
    seq_len = 1000
    seq_num = tensor2d.shape[0] // seq_len
    for j in range(min(tensor2d.shape[1], 100)):
        print(f'{j}/{tensor2d.shape[1]}')
        os.makedirs(f'{save_path}/channel_{j}', exist_ok=True)
        tensor1d = tensor2d[:, j]
        for i in range(seq_num):
            tensor_part = tensor1d[i*seq_len: (i+1)*seq_len]
            plot_sequence(tensor_part, f'{save_path}/channel_{j}/{i}')
        tensor_part = tensor1d[-1000:]
        plot_sequence(tensor_part, f'{save_path}/channel_{j}/{seq_num}')


class MultiDomainSelfSupervisionDataset(Dataset):
    def __init__(self, data_name='etth1', seq_len=336):
        self.transfer1 = MinMaxScaler(feature_range=(0, 1))
        self.transfer2 = StandardScaler()

        self.seq_len = seq_len
        self.dataset = []
        for flag in ['train', 'test', 'val']:
            dataset = np.load(f'./dataset/series/{flag}_dataset/{data_name}.npy')
            self.dataset.append(dataset)
        self.dataset = np.concatenate(self.dataset, axis=0)
        self.dataset = self.transfer2.fit_transform(self.dataset)
        self.dataset = torch.from_numpy(self.dataset).float()

    def __len__(self):
        return (self.dataset.shape[0] // self.seq_len) * self.dataset.shape[1]

    def __getitem__(self, index):
        sequence_id = int(index // self.dataset.shape[1])
        channel_id = int(index - sequence_id * self.dataset.shape[1])
        sequence_begin = sequence_id * self.seq_len
        sequence_end = sequence_begin + self.seq_len
        data = self.dataset[sequence_begin:sequence_end, channel_id]
        return data


dataset_len_bais = {
    'etth1': 0,
    'ettm1': 203,
    'etth2': 994,
    'ettm2': 1197,
    'electricity': 1988,
    'traffic': 18680,
    'pems': 48850,
}


class MultiDomainSelfSupervisionDataset_force(Dataset):
    def __init__(self, data_name='etth1', seq_len=336):
        self.transfer1 = MinMaxScaler(feature_range=(0, 1))
        self.transfer2 = StandardScaler()
        self.data_name = data_name

        self.seq_len = seq_len
        self.dataset = []
        for flag in ['train', 'test', 'val']:
            dataset = np.load(f'./dataset/series/{flag}_dataset/{data_name}.npy')
            self.dataset.append(dataset)
        self.dataset = np.concatenate(self.dataset, axis=0)
        self.dataset = self.transfer2.fit_transform(self.dataset)
        self.dataset = torch.from_numpy(self.dataset).float()

    def __len__(self):
        return (self.dataset.shape[0] // self.seq_len) * self.dataset.shape[1]

    def __getitem__(self, index):
        sequence_id = int(index // self.dataset.shape[1])
        channel_id = int(index - sequence_id * self.dataset.shape[1])
        sequence_begin = sequence_id * self.seq_len
        sequence_end = sequence_begin + self.seq_len
        data = self.dataset[sequence_begin:sequence_end, channel_id]
        out_dirt = {
            'data': data,
            'index': index + dataset_len_bais[self.data_name],
        }
        return out_dirt


class MultiDomainSelfSupervisionDataset_classification_tfc(Dataset):
    def __init__(self, data_name='FD-A'):
        self.transfer2 = StandardScaler()

        self.x_data = []
        for flag in ['train', 'test', 'val']:
            x_data = np.load(f'./dataset/series/_classification_tfc/{flag}_dataset/{data_name}_x_data.npy')[:, :, 0]
            self.x_data.append(x_data)
        self.x_data = np.concatenate(self.x_data, axis=0)
        self.x_data = self.transfer2.fit_transform(self.x_data)

        self.y_data = []
        for flag in ['train', 'test', 'val']:
            y_data = np.load(f'./dataset/series/_classification_tfc/{flag}_dataset/{data_name}_y_data.npy')
            self.y_data.append(y_data)
        self.y_data = np.concatenate(self.y_data, axis=0)

    def __len__(self):
        return self.x_data.shape[0]

    def __getitem__(self, index):
        x_data = self.x_data[index]
        y_data = self.y_data[index]
        out_dirt = {
            'x_data': x_data,
            'y_data': y_data,
        }
        return out_dirt


class MultiDomainSelfSupervisionDataset_classification_ucr(Dataset):
    def __init__(self, data_name='ACSF1', seq_len=1024):
        self.data_name = data_name
        self.transfer1 = MinMaxScaler(feature_range=(0, 1))
        self.transfer2 = StandardScaler()

        self.x_data = []
        for flag in ['train', 'test', 'val']:
            x_data = np.load(f'./dataset/series/_classification_ucr/{flag}_dataset/{data_name}_x_data.npy')[:, :, 0]
            self.x_data.append(x_data)
        self.x_data = np.concatenate(self.x_data, axis=0)
        self.x_data = self.transfer2.fit_transform(self.x_data)
        self.x_data = torch.from_numpy(self.x_data)
        self.x_data = self.x_data.unsqueeze(1)

        import torch.nn.functional as F
        self.x_data = F.interpolate(
            self.x_data,
            size=seq_len,
            mode='linear',
            align_corners=True
        )
        self.x_data = self.x_data.squeeze(1)

        self.y_data = []
        for flag in ['train', 'test', 'val']:
            y_data = np.load(f'./dataset/series/_classification_ucr/{flag}_dataset/{data_name}_y_data.npy')
            self.y_data.append(y_data)
        self.y_data = np.concatenate(self.y_data, axis=0)

    def __len__(self):
        return self.x_data.shape[0]

    def __getitem__(self, index):
        x_data = self.x_data[index]
        y_data = self.y_data[index]
        out_dirt = {
            'x_data': x_data,
            'y_data': y_data,
            'from': self.data_name,
        }
        return out_dirt


class UCR128LoaderTimeseriesAndImage(Dataset):
    def __init__(self, data_name='ArrowHead', seq_len=1024, flag_list=('train', 'test', 'val')):

        self.transfer1 = MinMaxScaler(feature_range=(0, 1))
        self.transfer2 = StandardScaler()

        self.x_data = []
        for flag in flag_list:
            x_data = np.load(f'./dataset/series/_classification_ucr/{flag}_dataset/{data_name}_x_data.npy')[:, :, 0]
            self.x_data.append(x_data)
        self.x_data = np.concatenate(self.x_data, axis=0)
        self.x_data = self.transfer2.fit_transform(self.x_data)
        self.x_data = torch.from_numpy(self.x_data)
        self.x_data = self.x_data.unsqueeze(1)

        import torch.nn.functional as F
        self.x_data = F.interpolate(
            self.x_data,
            size=seq_len,
            mode='linear',
            align_corners=True
        )
        self.x_data = self.x_data.squeeze(1)

        self.y_data = []
        for flag in flag_list:
            y_data = np.load(f'./dataset/series/_classification_ucr/{flag}_dataset/{data_name}_y_data.npy')
            self.y_data.append(y_data)
        self.y_data = np.concatenate(self.y_data, axis=0)
        self.class_num = len(set(self.y_data))

        self.label2qind = {}
        all_keys = list(target_index.keys())
        for i in range(self.class_num):
            q_inds_idx = all_keys[i]
            q_inds_1000 = torch.from_numpy(np.load(f'yucornetto/_pretrained_discrete_space_1000/class_{q_inds_idx}_discrete_space.npy')).long()
            q_ind = q_inds_1000[best_index[q_inds_idx], :]
            self.label2qind[i] = q_ind

    def __getitem__(self, index):
        x_data = self.x_data[index]
        y_data = self.y_data[index]
        q_ind = self.label2qind[int(y_data)]

        out_dirt = {
            'index': index,
            'x_data': x_data,
            'y_data': y_data,
            'q_ind': q_ind,
        }
        return out_dirt

    def __len__(self):
        return self.x_data.shape[0]


target_index = {
    948: '青苹果', 950: '橙子', 951: '柠檬', 954: '香蕉', 957: '石榴', 985: '小雏菊', 945: '彩椒',
    619: '床头灯', 736: '台球', 779: '校车', 805: '足球', 820: '火车', 849: '紫砂壶', 873: '凯旋门', 604: '沙漏', 607: '南瓜灯笼',
    31: '树蛙', 130: '火烈鸟', 132: '白鹤', 145: '企鹅', 148: '虎鲸',
    301: '七星瓢虫', 311: '蟋蟀', 323: '蝴蝶', 327: '海星', 333: '金丝熊鼠鼠', 339: '马', 340: '斑马', 388: '大熊猫', 393: '小丑鱼',
    402: '大提琴', 409: '钟表', 417: '热气球', 429: '棒球', 437: '灯塔', 440: '洋酒', 448: '鸟屋', 449: '水上木屋', 468: '黄色出租车',
    470: '蜡烛', 478: '纸箱', 497: '城堡', 508: '键盘', 524: '金属盔甲', 574: '高尔夫球', 579: '钢琴', 643: '面具', 852: '网球',
    927: '蛋糕', 932: '甜甜圈', 960: '巧克力冰淇淋', 967: '咖啡',
    179: '斯塔福郡斗牛犬', 207: '金毛', 232: '边牧', 235: '德牧', 258: '萨摩耶', 259: '博美犬', 263: '柯基', 284: '暹罗猫', 250: '西伯利亚雪橇犬(哈士奇)',
    506: '旋转楼梯', 518: '摩托头盔', 971: '泡泡',
}

best_index = {
    948: 66, 950: 1, 951: 99, 954: 78, 957: 11, 985: 0, 945: 3,
    619: 3, 736: 8, 779: 27, 805: 82, 820: 4, 849: 3, 873: 16, 604: 6, 607: 11,
    31: 37, 130: 6, 132: 13, 145: 5, 148: 8,
    301: 11, 311: 12, 323: 18, 327: 39, 333: 48, 339: 52, 340: 46, 388: 67, 393: 96,
    402: 51, 409: 31, 417: 37, 429: 43, 437: 24, 440: 29, 448: 33, 449: 36, 468: 40,
    470: 25, 478: 17, 497: 23, 508: 38, 524: 49, 574: 19, 579: 18, 643: 75, 852: 32,
    927: 28, 932: 0, 960: 39, 967: 2,
    179: 0, 207: 12, 232: 17, 235: 46, 258: 37, 259: 55, 263: 26, 284: 85, 250: 99,
    506: 5, 518: 17, 971: 5,
}

all_target_index = {
    7: '公鸡', 11: '黄鸟', 13: '雪中灰鸟', 14: '蓝鸟', 22: '老鹰', 31: '树蛙', 42: '蓝红蜥蜴', 96: '大嘴鹦鹉', 100: '天鹅',
    101: '大象', 105: '树獭', 113: '蜗牛', 122: '麻辣小龙虾', 130: '火烈鸟', 132: '白鹤', 145: '企鹅', 148: '虎鲸',
    151: '吉娃娃', 153: '马尔济斯', 156: '查尔斯王小猎犬', 157: '蝴蝶犬', 160: '罗得西亚脊背犬', 166: '比格犬', 179: '斯塔福郡斗牛犬',
    185: '诺福克梗', 195: '波士顿梗', 207: '金毛', 208: '拉布拉多犬', 217: '英国史宾格犬', 225: '马里努阿犬', 232: '边牧', 234: '罗威纳犬',
    235: '德牧', 240: '大瑞士山地犬', 249: '阿拉斯加雪橇犬', 250: '西伯利亚雪橇犬(哈士奇)', 258: '萨摩耶', 259: '博美犬', 263: '柯基', 265: '泰迪', 268: '佐罗兹英特利犬',
    277: '狐狸', 284: '暹罗猫', 290: '猎豹', 291: '狮子', 292: '老虎', 294: '棕熊', 295: '黑熊', 301: '七星瓢虫', 309: '蜜蜂', 311: '蟋蟀',
    323: '蝴蝶', 327: '海星', 333: '金丝熊鼠鼠', 339: '马', 340: '斑马', 342: '野猪', 344: '河马', 350: '羚羊', 354: '骆驼',
    355: '羊驼', 366: '大猩猩', 387: '小浣熊', 388: '大熊猫', 393: '小丑鱼', 402: '大提琴', 404: '飞机', 407: '救护车', 417: '热气球',
    421: '楼梯', 423: '座椅', 427: '木制酒桶', 429: '棒球', 436: '汽车', 437: '灯塔', 440: '洋酒', 441: '啤酒', 966: '红酒', 442: '钟楼',
    447: '望远镜', 448: '鸟屋', 449: '水上木屋', 453: '书柜', 455: '瓶盖', 460: '海堤', 468: '黄色出租车', 470: '蜡烛', 471: '火炮',
    478: '纸箱', 497: '城堡', 504: '水杯', 506: '旋转楼梯', 508: '键盘', 514: '靴子', 515: '牛仔帽', 518: '摩托头盔', 519: '木箱',
    521: '电饭煲', 524: '金属盔甲', 527: '电脑', 528: '电话', 532: '桌椅', 541: '架子鼓', 545: '风扇', 546: '吉他', 555: '消防车',
    562: '喷泉', 568: '毛皮大衣', 572: '玻璃高脚杯', 574: '高尔夫球', 579: '钢琴', 603: '马车', 604: '沙漏', 607: '南瓜灯笼', 619: '床头灯',
    628: '轮船', 643: '面具', 649: '复活节岛石像', 718: '桥梁', 719: '猪猪存钱罐', 736: '台球', 751: '赛车', 779: '校车', 780: '帆船',
    805: '足球', 820: '火车', 821: '桥梁', 849: '紫砂壶', 850: '泰迪熊', 852: '网球', 873: '凯旋门', 883: '花瓶', 917: '漫画',
    927: '蛋糕', 928: '冰淇淋', 960: '巧克力冰淇淋', 967: '咖啡', 930: '面包', 932: '甜甜圈', 934: '热狗', 963: '披萨', 937: '西兰花',
    939: '黄瓜', 945: '彩椒', 948: '青苹果', 949: '草莓', 950: '橙子', 951: '柠檬', 954: '香蕉', 957: '石榴', 971: '泡泡', 525: '水坝',
    970: '雪山', 972: '悬崖', 975:'湖边', 978: '海边', 979: '山谷', 980: '火山', 985: '小雏菊', 987: '玉米', 990: '栗子'
}


def test_x_data_and_q_ind():
    dataset = UCR128LoaderTimeseriesAndImage()
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=8, num_workers=0, shuffle=True)

    dirt_x = {}
    dirt_q = {}
    for batch in dataloader:
        x_data = batch['x_data']
        q_ind = batch['q_ind']
        index = batch['index']
        for i, ind in enumerate(index.numpy().tolist()):
            dirt_x[ind] = x_data[i, :]
            dirt_q[ind] = q_ind[i, :]
    print(len(dirt_x))
    print(len(dirt_q))

    for batch in dataloader:
        x_data = batch['x_data']
        q_ind = batch['q_ind']
        index = batch['index']
        for i, ind in enumerate(index.numpy().tolist()):
            print('')
            print(sum(dirt_x[ind] - x_data[i, :]))
            print(sum(dirt_q[ind] - q_ind[i, :]))


if __name__ == '__main__':

    dataset_list = []
    for data_name in ['etth1', 'etth2', 'ettm1', 'ettm2', 'electricity', 'traffic', 'pems']:
        dataset = MultiDomainSelfSupervisionDataset(data_name=data_name, seq_len=512)
        dataset_list.append(dataset)
    concat_dataset = ConcatDataset(dataset_list)

    dataset_list = []
    for data_name in ['FD-A', 'FD-B']:
        dataset = MultiDomainSelfSupervisionDataset_classification_tfc(data_name=data_name)
        dataset_list.append(dataset)
    concat_dataset = ConcatDataset(dataset_list)

    load_path = f'./dataset/series/_classification_ucr/train_dataset'
    tmp_list = os.listdir(load_path)
    data_name_list = []
    for data_name in tmp_list:
        data_name_list.append(data_name.split('_')[0])
    data_name_list = list(set(data_name_list))
    dataset_list = []
    for data_name in data_name_list:
        if 'AllGestureWiimote' not in data_name:
            dataset = MultiDomainSelfSupervisionDataset_classification_ucr(data_name=data_name)
            dataset_list.append(dataset)
    concat_dataset = ConcatDataset(dataset_list)

    load_path = 'dataset/_UCR_uni'
    data_name_list = ['SyntheticControl', 'FacesUCR', 'GunPointMaleVersusFemale', 'GunPointOldVersusYoung']
    for data_name in data_name_list:
        dataset = UCR128LoaderTimeseriesAndImage(dataroot=load_path, root_path=data_name, flag='train')



        

        
        


        
        
        



