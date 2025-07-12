import os
import torch
import numpy as np
import pandas as pd

from scipy import io
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from Models.interpretable_diffusion.model_utils import normalize_to_neg_one_to_one, unnormalize_to_zero_to_one
from Utils.masking_utils import noise_mask


class CustomDataset(Dataset):
    def __init__(
        self, 
        name,
        data_root, 
        seq_len=64, 
        proportion=0, 
        save2npy=True, 
        neg_one_to_one=True,
        seed=123,
        period='train',
        mode = None,
        output_dir='./OUTPUT',
        pred_len=None,
        missing_ratio=None,
        style='separate', 
        distribution='geometric', 
        mean_mask_length=3
    ):
        super(CustomDataset, self).__init__()
        print('dataset seq_len, pred_len , missing_ratio = ', seq_len, pred_len, missing_ratio)
        assert period in ['train', 'test'], 'period must be train or test.'
        if period == 'train':
            assert ~(pred_len is not None or missing_ratio is not None), ''
        self.name, self.pred_len, self.missing_ratio = name, pred_len, missing_ratio
        self.style, self.distribution, self.mean_mask_length = style, distribution, mean_mask_length
        self.rawdata, self.scaler = self.read_data(data_root, self.name)
        self.dir = os.path.join(output_dir, 'samples')
        os.makedirs(self.dir, exist_ok=True)

        self.seq_len, self.period = seq_len, period
        self.len, self.var_num = self.rawdata.shape[0], self.rawdata.shape[-1]
        self.save2npy = save2npy
        self.auto_norm = neg_one_to_one

        self.data = self.__normalize(self.rawdata)
        if proportion>0:
            train, inference = self.__getsamples(self.data, proportion, seed)
            self.samples = train if period == 'train' else inference
        else:
            self.samples, self.targets = self.__getwindows(self.data, seq_len, pred_len)
            print(self.samples.shape, 'samples shape ...')
        
        
        if missing_ratio is not None and missing_ratio < 1:
            print('mode === ', mode, 'missing_ratio = ', missing_ratio)
            self.masking = self.mask_data(seed)
        elif pred_len is not None:
            print('mode === ', mode, 'pred_len = ', pred_len)
            masks = np.ones(self.samples.shape)
            self.masking = masks.astype(bool)
        else:
            raise NotImplementedError()
            
        self.sample_num = self.samples.shape[0]

    def __getwindows(self, data, data_len, pred_len):
        sample_num_total = max(self.len - data_len - pred_len + 1, 0)
        x = np.zeros((sample_num_total, data_len, self.var_num))
        y = np.zeros((sample_num_total, pred_len, self.var_num))
        for i in range(sample_num_total):
            start = i
            end = i + data_len
            x[i, :, :] = data[start:end, :]
            y[i, :, :] = data[end:end+pred_len, :]
        return x, y

    def __getsamples(self, data, proportion, seed):
        x = np.zeros((self.sample_num_total, self.seq_len, self.var_num))
        for i in range(self.sample_num_total):
            start = i
            end = i + self.seq_len
            x[i, :, :] = data[start:end, :]

        train_data, test_data = self.divide(x, proportion, seed)
        
        print(train_data.shape, test_data.shape, 'train_data and test_data shape ...')

        if self.save2npy:
            if 1 - proportion > 0:
                np.save(os.path.join(self.dir, f"{self.name}_ground_truth_{self.seq_len}_test.npy"), self.unnormalize(test_data))
            np.save(os.path.join(self.dir, f"{self.name}_ground_truth_{self.seq_len}_train.npy"), self.unnormalize(train_data))
            if self.auto_norm:
                if 1 - proportion > 0:
                    np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.seq_len}_test.npy"), unnormalize_to_zero_to_one(test_data))
                np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.seq_len}_train.npy"), unnormalize_to_zero_to_one(train_data))
            else:
                if 1 - proportion > 0:
                    np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.seq_len}_test.npy"), test_data)
                np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.seq_len}_train.npy"), train_data)

        return train_data, test_data

    def normalize(self, sq):
        d = sq.reshape(-1, self.var_num)
        d = self.scaler.transform(d)
        if self.auto_norm:
            d = normalize_to_neg_one_to_one(d)
        return d.reshape(-1, self.seq_len, self.var_num)

    def unnormalize(self, sq):
        d = self.__unnormalize(sq.reshape(-1, self.var_num))
        return d.reshape(-1, self.seq_len, self.var_num)
    
    def __normalize(self, rawdata):
        data = self.scaler.transform(rawdata)
        if self.auto_norm:
            data = normalize_to_neg_one_to_one(data)
        return data

    def __unnormalize(self, data):
        if self.auto_norm:
            data = unnormalize_to_zero_to_one(data)
        x = data
        return self.scaler.inverse_transform(x)
    
    @staticmethod
    def divide(data, ratio, seed=2024):
        size = data.shape[0]
        # Store the state of the RNG to restore later.
        st0 = np.random.get_state()
        np.random.seed(seed)

        regular_train_num = int(np.ceil(size * ratio))
        id_rdm = np.random.permutation(size)
        regular_train_id = id_rdm[:regular_train_num]
        irregular_train_id = id_rdm[regular_train_num:]

        regular_data = data[regular_train_id, :]
        irregular_data = data[irregular_train_id, :]

        # Restore RNG.
        np.random.set_state(st0)
        return regular_data, irregular_data

    @staticmethod
    def read_data(filepath, name=''):
        """Reads a single .csv
        """
        print(name,'dataset name: ' + filepath)
        if 'C2TM' in name:
            df = pd.read_csv(filepath)/(8*1024*1024)
            df = df.drop('time',axis=1)
            data = df.values
        elif 'CBS' in name:
            df = pd.read_csv(filepath)
            df = df.drop('time',axis=1)
            data = df.values
        elif 'SMS-IN' in name:
            data = np.load(filepath)[...,0]/10
            print(data.shape, 'SMS-IN...')
        elif 'SMS-OUT' in name:
            data = np.load(filepath)[...,1]/10
            print(data.shape, 'SMS-OUT...')
        elif 'Call-IN' in name:
            data = np.load(filepath)[...,2]/10
            print(data.shape, 'Call IN...')
        elif 'Call-OUT' in name:
            data = np.load(filepath)[...,3]/10
            print(data.shape, 'Call OUT...')
        elif 'Internet' in name: 
            data = np.load(filepath)[...,-1]/10
            print(data.shape, 'Internet ...')
        elif 'Milano' in name:
            data = np.load(filepath)/10
            print(data.shape, 'Milano ...')
        else:
            df = pd.read_csv(filepath, header=0)
            if name == 'etth':
                df.drop(df.columns[0], axis=1, inplace=True)
            data = df.values
        
        scaler = MinMaxScaler()
        scaler = scaler.fit(data)
        
        # print(data.shape, 'data shape')
        # B, N, C = data.shape
        # data = data.reshape(B, -1)
        # scaler = MinMaxScaler()
        # scaler = scaler.fit(data)
        # data = data.reshape(B, N, C)
        return data, scaler
    
    def make_mask(self, seed=2024):
        masks = np.ones_like(self.samples)
        # Store the state of the RNG to restore later.
        st0 = np.random.get_state()
        np.random.seed(seed)

        for idx in range(self.samples.shape[0]):
            x = self.samples[idx, :, :]  # (seq_length, feat_dim) array
            mask = noise_mask(x, self.missing_ratio, self.mean_mask_length, self.style,
                              self.distribution)  # (seq_length, feat_dim) boolean array
            masks[idx, :, :] = mask

        if self.save2npy:
            np.save(os.path.join(self.dir, f"{self.name}_masking_{self.seq_len}.npy"), masks)

        # Restore RNG.
        np.random.set_state(st0)
        return masks.astype(bool)

    # def __getitem__(self, ind):
    #     if self.period == 'test':
    #         x = self.samples[ind, :, :]  # (seq_length, feat_dim) array
    #         m = self.masking[ind, :, :]  # (seq_length, feat_dim) boolean array
    #         y = self.targets[ind, :, :] 
    #         return torch.from_numpy(x).float(), torch.from_numpy(y).float(), torch.from_numpy(m)
    #     x = self.samples[ind, :, :]  # (seq_length, feat_dim) array
    #     y = self.targets[ind, :, :] 
    #     return torch.from_numpy(x).float(), torch.from_numpy(y).float()
    
    def __getitem__(self, ind):
        x = self.samples[ind, :, :]  # (seq_length, feat_dim) array
        y = self.targets[ind, :, :] 
        m = self.masking[ind, :, :]  # (seq_length, feat_dim) boolean array
        return torch.from_numpy(x).float(), torch.from_numpy(y).float(), torch.from_numpy(m)

    def __len__(self):
        return self.sample_num
    

class fMRIDataset(CustomDataset):
    def __init__(
        self, 
        proportion=1., 
        **kwargs
    ):
        super().__init__(proportion=proportion, **kwargs)

    @staticmethod
    def read_data(filepath, name=''):
        """Reads a single .csv
        """
        data = io.loadmat(filepath + '/sim4.mat')['ts']
        scaler = MinMaxScaler()
        scaler = scaler.fit(data)
        return data, scaler
