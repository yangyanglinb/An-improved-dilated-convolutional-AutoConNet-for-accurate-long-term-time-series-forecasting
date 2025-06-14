import torch
import os, warnings
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features

warnings.filterwarnings('ignore')


# ============================================================
# 0.  公共小工具
# ============================================================
def _split_by_ratio(n_total, train_ratio=0.6, test_ratio=0.2):
    """返回 [(train_start,train_end), (val_start,val_end), (test_start,test_end)]"""
    n_train = int(n_total * train_ratio)
    n_test = int(n_total * test_ratio)
    n_val = n_total - n_train - n_test
    return [
        (0, n_train),
        (n_train, n_train + n_val),
        (n_total - n_test, n_total)
    ]


def _build_time_stamp(series, freq, enc_type):
    """根据 enc_type 返回时间特征矩阵"""
    if enc_type == 0:  # 手工拆分
        df = pd.DataFrame({'date': series})
        df['month'] = df.date.dt.month
        df['day'] = df.date.dt.day
        df['weekday'] = df.date.dt.weekday
        df['hour'] = df.date.dt.hour
        if freq == 't':  # minute 级再给 minute
            df['minute'] = df.date.dt.minute // 15
        return df.drop('date', axis=1).values
    else:  # 官方函数
        return time_features(pd.DatetimeIndex(series), freq=freq).transpose(1, 0)


def _auto_adjust_window(obj, n_total):
    """
    若原 seq_len 太大，自动压缩：
      new_seq_len = min(seq_len, n_total - pred_len - 1)
      new_label_len ≤ new_seq_len
    调整后保证 __len__() ≥ 1
    """
    max_seq = n_total - obj.pred_len - 1
    if max_seq <= 0:
        obj.seq_len = max(1, n_total - obj.pred_len)
    else:
        obj.seq_len = min(obj.seq_len, max_seq)

    obj.label_len = min(obj.label_len, obj.seq_len)


# ============================================================
# 1.  ETT ‒ Hourly
# ============================================================
class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 seasonal_patterns=None, train_ratio=0.6):

        # 如果 size=None，就使用默认 (96, 48, 96)
        self.seq_len, self.label_len, self.pred_len = size or (96, 48, 96)
        assert flag in ['train', 'val', 'test']
        self.set_type = {'train': 0, 'val': 1, 'test': 2}[flag]

        self.features, self.target = features, target
        self.scale, self.timeenc, self.freq = scale, timeenc, freq.lower()

        self.root_path, self.data_path = root_path, data_path
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        borders = [
            (0, 12 * 30 * 24),
            (12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24),
            (12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24)
        ]
        b1, b2 = borders[self.set_type]

        # 提取原始数值数据列
        if self.features in ('M', 'MS'):
            df_data = df_raw.iloc[:, 1:]
        else:
            df_data = df_raw[[self.target]]
        # 只保留数值列，剔除所有非数值列
        df_data = df_data.select_dtypes(include=[np.number])
        # 再次强制转换并填 NA
        df_data = df_data.apply(pd.to_numeric, errors='coerce').fillna(0)

        # 归一化
        if self.scale:
            scaler = StandardScaler()
            scaler.fit(df_data.iloc[borders[0][0]:borders[0][1]].values)
            data = scaler.transform(df_data.values)
            self.scaler = scaler
        else:
            data = df_data.values

        # 构造时间戳特征
        df_stamp = pd.to_datetime(df_raw['date'].iloc[b1:b2])
        data_stamp = _build_time_stamp(df_stamp, self.freq, self.timeenc)

        self.data_x = data[b1:b2]
        self.data_y = self.data_x
        self.data_stamp = data_stamp

        _auto_adjust_window(self, len(self.data_x))

    def __getitem__(self, idx):
        s, e = idx, idx + self.seq_len
        r0 = e - self.label_len
        r1 = r0 + self.label_len + self.pred_len
        return (
            idx,
            self.data_x[s:e],
            self.data_y[r0:r1],
            self.data_stamp[s:e],
            self.data_stamp[r0:r1]
        )

    def __len__(self):
        return max(1, len(self.data_x) - self.seq_len - self.pred_len + 1)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


# ============================================================
# 2.  ETT ‒ Minute
# ============================================================
class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t',
                 seasonal_patterns=None, train_ratio=0.6):

        self.seq_len, self.label_len, self.pred_len = size or (96, 48, 96)
        assert flag in ['train', 'val', 'test']
        self.set_type = {'train': 0, 'val': 1, 'test': 2}[flag]

        self.features, self.target = features, target
        self.scale, self.timeenc, self.freq = scale, timeenc, freq.lower()

        self.root_path, self.data_path = root_path, data_path
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        borders = [
            (0, 12 * 30 * 24 * 4),
            (12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4),
            (12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4)
        ]
        b1, b2 = borders[self.set_type]

        if self.features in ('M', 'MS'):
            df_data = df_raw.iloc[:, 1:]
        else:
            df_data = df_raw[[self.target]]
        df_data = df_data.select_dtypes(include=[np.number])
        df_data = df_data.apply(pd.to_numeric, errors='coerce').fillna(0)

        if self.scale:
            scaler = StandardScaler()
            scaler.fit(df_data.iloc[borders[0][0]:borders[0][1]].values)
            data = scaler.transform(df_data.values)
            self.scaler = scaler
        else:
            data = df_data.values

        df_stamp = pd.to_datetime(df_raw['date'].iloc[b1:b2])
        data_stamp = _build_time_stamp(df_stamp, self.freq, self.timeenc)

        self.data_x = data[b1:b2]
        self.data_y = self.data_x
        self.data_stamp = data_stamp

        _auto_adjust_window(self, len(self.data_x))

    def __getitem__(self, idx):
        s, e = idx, idx + self.seq_len
        r0 = e - self.label_len
        r1 = r0 + self.label_len + self.pred_len
        return (
            idx,
            self.data_x[s:e],
            self.data_y[r0:r1],
            self.data_stamp[s:e],
            self.data_stamp[r0:r1]
        )

    def __len__(self):
        return max(1, len(self.data_x) - self.seq_len - self.pred_len + 1)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


# ============================================================
# 3.  通用 CSV
# ============================================================
class Dataset_Custom(Dataset):
    """
    任意 CSV：
      - 若无 date 列自动生成；
      - 支持 h/t/s/d/w/m/q/y 频率；
    """
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='data.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 seasonal_patterns=None, train_ratio=0.6):

        self.seq_len, self.label_len, self.pred_len = size or (96, 48, 96)
        assert flag in ['train', 'val', 'test']
        self.set_type = {'train': 0, 'val': 1, 'test': 2}[flag]

        self.features, self.target = features, target
        self.scale, self.timeenc, self.freq = scale, timeenc, freq.lower()
        self.train_ratio = train_ratio

        self.root_path, self.data_path = root_path, data_path
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # ----- date 列保障 -----
        cols = list(df_raw.columns)
        if self.target in cols:
            cols.remove(self.target)
        if 'date' in cols:
            cols.remove('date')
        else:
            freq_map = {'h': 'H', 't': 'T', 's': 'S',
                        'd': 'D', 'w': 'W',
                        'm': 'M', 'q': 'Q', 'y': 'A'}
            step = freq_map.get(self.freq, 'D')
            try:
                df_raw.insert(0, 'date',
                              pd.date_range('2000-01-01', periods=len(df_raw), freq=step))
            except Exception as e:
                print('[warn] date_range failed:', e, '— use int index')
                df_raw.insert(0, 'date',
                              pd.to_datetime(range(len(df_raw)), unit='D'))

        df_raw = df_raw[['date'] + cols + [self.target]]

        # ----- split -----
        borders = _split_by_ratio(len(df_raw), self.train_ratio, 0.2)
        b1, b2 = borders[self.set_type]

        # ----- 数值 -----
        if self.features in ('M', 'MS'):
            df_data = df_raw.iloc[:, 1:]
        else:
            df_data = df_raw[[self.target]]
        df_data = df_data.select_dtypes(include=[np.number])
        df_data = df_data.apply(pd.to_numeric, errors='coerce').fillna(0)

        if self.scale:
            scaler = StandardScaler()
            scaler.fit(df_data.iloc[borders[0][0]:borders[0][1]].values)
            data = scaler.transform(df_data.values)
            self.scaler = scaler
        else:
            data = df_data.values

        # ----- 时间 -----
        df_stamp = pd.to_datetime(df_raw['date'])
        data_stamp = _build_time_stamp(df_stamp, self.freq, self.timeenc)

        # ----- 保存 -----
        self.data_x = data[b1:b2]
        self.data_y = self.data_x
        self.data_stamp = data_stamp[b1:b2]

        _auto_adjust_window(self, len(self.data_x))

    def __getitem__(self, idx):
        s, e = idx, idx + self.seq_len
        r_begin = e - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        return (
            idx,
            self.data_x[s:e],
            self.data_y[r_begin:r_end],
            self.data_stamp[s:e],
            self.data_stamp[r_begin:r_end]
        )

    def __len__(self):
        return max(1, len(self.data_x) - self.seq_len - self.pred_len + 1)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
