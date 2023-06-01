from torch.utils.data import Dataset

from os import path
import warnings
from abc import ABC, abstractmethod
from typing import Sequence, Optional, Dict
import os 
import pickle
import numpy as np
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
warnings.filterwarnings('ignore')






class DatasetTSBase(ABC, Dataset):
    # root_path --> Dataset dir
    # flag      --> ["train", "test", "val"]
    # size      --> [seq_len, label_len, pred_len] => [lookback, overlap, horizon]
    # features  --> [M, S] => [multivariate, siglevariate]
    # data_path --> relative data file path
    # target    --> target column
    # scale     --> flag to apply scaling
    def __init__(
        self,
        root_path: str,
        flag: str = "train",
        size: Sequence = None,
        features: str = "M",
        data_path: str = "ETTh1.csv",
        target: str = "OT",
        scale: bool = True,
    ):
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len, self.label_len, self.pred_len = size

        # init
        assert flag in ['train', 'test', 'val']
        self.flag = flag

        # M - multivariate; S - univariate
        assert features in ['M', 'S']
        self.features = features
        self.target = target
        self.scale = scale

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    @abstractmethod
    def _get_borders(self, data_len: Optional[int]) -> Dict[str, slice]:
        pass

    def _filter_data_by_columns(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        if self.features == 'M':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        return df_data

    def _get_scaled_data(self, df_data: pd.DataFrame) -> np.ndarray:
        train_data = df_data[self._get_borders(len(df_data))["train"]]
        self.scaler.fit(train_data.values)
        return self.scaler.transform(df_data.values)

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(path.join(self.root_path, self.data_path))

        data_indices = self._get_borders(len(df_raw))[self.flag]

        df_data = self._filter_data_by_columns(df_raw)
        data = self._get_scaled_data(
            df_data) if self.scale else df_data.values

        self.data_x = data[data_indices]
        self.data_y = data[data_indices]

    def __getitem__(self, index):
        in_slice = slice(index, index + self.seq_len)
        pred_start = in_slice.stop - self.label_len
        out_slice = slice(pred_start, pred_start + self.pred_len)

        seq_x = self.data_x[in_slice]
        seq_y = self.data_y[out_slice]

        return seq_x, seq_y

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class DatasetETTHour(DatasetTSBase):
    def __init__(
        self,
        root_path: str,
        flag: str = 'train',
        size: Sequence = None,
        features: str = 'M',
        data_path: str = 'ETTh1.csv',
        target: str = 'OT',
        scale: bool = True
    ):
        super().__init__(root_path, flag, size, features, data_path, target, scale)

    def _get_borders(self, data_len: Optional[int]) -> Dict[str, slice]:
        month = 30 * 24
        return {
            "train": slice(0, 12 * month),
            "val": slice(12 * month - self.seq_len, 16 * month),
            "test": slice(16 * month - self.seq_len, 20 * month)
        }


class DatasetETTMinute(DatasetTSBase):
    def __init__(
        self,
        root_path: str,
        flag: str = 'train',
        size: Sequence = None,
        features: str = 'M',
        data_path: str = 'ETTm1.csv',
        target: str = 'OT',
        scale: bool = True,
    ):
        super().__init__(root_path, flag, size, features, data_path, target, scale)

    def _get_borders(self, data_len: Optional[int]) -> Dict[str, slice]:
        month = 30 * 24 * 4
        return {
            "train": slice(0, 12 * month),
            "val": slice(12 * month - self.seq_len, 16 * month),
            "test": slice(16 * month - self.seq_len, 20 * month)
        }


class DatasetCustom(DatasetTSBase):
    def __init__(
        self,
        root_path: str,
        flag: str = 'train',
        size: Sequence = None,
        features: str = 'M',
        data_path: str = 'ETTh1.csv',
        target: str = 'OT',
        scale: bool = True
    ):
        super().__init__(root_path, flag, size, features, data_path, target, scale)

    def _get_borders(self, data_len: Optional[int]) -> Dict[str, slice]:
        num_train = int(data_len * 0.7)
        num_test = int(data_len * 0.2)
        num_vali = data_len - num_train - num_test

        return {
            "train": slice(0, num_train),
            "val": slice(num_train - self.seq_len, num_train + num_vali),
            "test": slice(num_test - self.seq_len, data_len)
        }


class DatasetPred(DatasetTSBase):
    def __init__(
        self,
        root_path: str,
        flag: str = 'pred',
        size: Sequence = None,
        features: str = 'M',
        data_path: str = 'ETTm1.csv',
        target: str = 'OT',
        scale: bool = True,
        inverse: bool = False,
        cols: Optional[list] = None
    ):
        assert flag in ['pred']
        self.inverse = inverse
        self.cols = cols

        super().__init__(root_path, flag, size, features,
                         data_path, target, scale)

    def __reorder_df_cols(self, df_raw: pd.DataFrame) -> list:
        if self.cols is not None:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        return df_raw[['date'] + cols + [self.target]]

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(path.join(self.root_path, self.data_path))

        data_indices = self._get_borders(len(df_raw))[self.flag]
        df_raw = self.__reorder_df_cols(df_raw)

        df_data = self._filter_data_by_columns(df_raw)
        data = self._get_scaled_data(
            df_data) if self.scale else df_data.values

        self.data_x = data[data_indices]
        if self.inverse:
            self.data_y = data[data_indices.stop: data_indices.start]
        else:
            self.data_y = data[data_indices]

    def __getitem__(self, index):
        in_slice = slice(index, index + self.seq_len)
        pred_start = in_slice.stop - self.label_len
        out_slice = slice(pred_start, pred_start + self.pred_len)

        seq_x = self.data_x[in_slice]
        if self.inverse:
            seq_y = self.data_y[out_slice.stop: out_slice.start]
        else:
            seq_y = self.data_y[out_slice]

        return seq_x, seq_y


def get_timeseries_datasets(
    root_path: str,
    size: Sequence[int],
    features: str,
    data_path: str,
    target: str = "OT",
):
    train_dataset = DatasetCustom(
        root_path, 'train', size, features, data_path, target
    )
    valid_dataset = DatasetCustom(
        root_path, 'val', size, features, data_path, target
    )
    info = {}

    return train_dataset, valid_dataset, info


class TimeseriesLMDataset(Dataset):
    
    def __init__(self, 
                 data_file: str,
                 return_tensor=False):
        data_file = os.path.join(data_file, 'electricity', 'processed_data_25_len_16_algo_kmean.pkl')
            
        with open(data_file, "rb") as f:
            proto_id, mean, std, data = pickle.load(f)
            
        self.proto_ids = proto_id
        self.mean = mean
        self.std = std
        self.data = data
        self.context_size = 50
        self.return_tensor = return_tensor
        
    def __getitem__(self, index):
        if self.return_tensor:
            return {"input_ids": torch.tensor(self.proto_ids[index: index + self.context_size]),}
        else:
            return {"input_ids": self.proto_ids[index: index + self.context_size],}
            
    
    def __len__(self):
        return len(self.proto_ids) - self.context_size
        
        
class PSBMDataset(Dataset):
    
    def __init__(self, 
                 data_file: str,
                 return_tensor=False):
        
        data_file = os.path.join(data_file, 'electricity', 'processed_data_25_len_16_algo_kmean.pkl')
        with open(data_file, "rb") as f:
            proto_id, mean, std, data = pickle.load(f)
            
        self.proto_ids = proto_id
        self.mean = mean
        self.std = std
        self.data = data
        self.context_size = 50
        self.return_tensor = return_tensor
        
    def __getitem__(self, index):
        if self.return_tensor:
            return {"input_ids": torch.tensor(self.proto_ids[index: index + self.context_size]),
                    'input_scale': torch.tensor(self.std[index: index + self.context_size].astype(np.float32)),
                    'input_bias': torch.tensor(self.mean[index: index + self.context_size].astype(np.float32)),
                    }
        else:
            return {"input_ids": self.proto_ids[index: index + self.context_size],
                'input_scale': self.std[index: index + self.context_size].astype(np.float32),
                'input_bias': self.mean[index: index + self.context_size].astype(np.float32),
                }
    def __len__(self):
        return len(self.proto_ids) - self.context_size
        


if __name__ == "__main__":
    train, valid, info = get_timeseries_datasets('data', None, 'M', 'ETTm1.csv')
    print(type(train[0][0]), type(train[0][1]), train[0][0].size, train[0][1].size)
    print(len(train))
    print(len(train[0]))