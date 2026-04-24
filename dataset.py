import os
import random
import torch
import numpy as np
import h5py
from tqdm import tqdm

class Dataset(torch.utils.data.Dataset):
    def __init__(self, exp_conf, data_dir, dataset_type, init_loc=None):
        assert dataset_type in ['train', 'test']
        self.exp_conf = exp_conf
        self.data_dir = data_dir
        self.dataset_path = os.path.join(self.data_dir, f'{dataset_type}.h5')
        assert os.path.exists(self.dataset_path)
        
        with h5py.File(self.dataset_path, 'r') as dataset:
            data_loc = dataset['data_loc'][:]
        self.num_data = self.exp_conf[f'num_{dataset_type}_data']
        assert self.num_data <= data_loc.shape[0]
        self.data_loc = torch.as_tensor(data_loc[:self.num_data], dtype=torch.float32) # [num_data, 3]
        self.dataset = None
        
        scheme = self.exp_conf['scheme']
        task = self.exp_conf['task']
        self.as_scenario_info = (scheme in ['alloc', 'iclloc'] and dataset_type == 'train')
        self.use_seq_dataset = (scheme in ['alloc', 'iclloc'] and task == 'train')
        self.gen_seq_datasets = (self.use_seq_dataset and self.exp_conf['gen_seq_datasets'])
        if self.use_seq_dataset:
            if self.gen_seq_datasets:
                self.seq_dataset_path = None
                self.seq_dataset = None
            else:
                self.train_dataset_obj = None
        
        if init_loc is not None:
            assert init_loc.shape[0] == self.num_data
            self.init_loc = torch.as_tensor(init_loc, dtype=torch.float32)
        else:
            self.init_loc = self.data_loc.clone()

    def __getitem__(self, index):
        '''
        Return
        data: [num_ant, num_car] or [max_len_nei + 1, num_ant, num_car], complex
        loc, init_loc: [2] or [max_len_nei, 2], float
        '''
        if self.use_seq_dataset and self.gen_seq_datasets:
            assert self.seq_dataset_path is not None
            if self.seq_dataset is None:
                self.seq_dataset = h5py.File(self.seq_dataset_path, 'r')
            data = torch.as_tensor(self.seq_dataset['data'][index], dtype=torch.complex64) # [max_len_nei + 1, num_ant, num_car]
            loc = torch.as_tensor(self.seq_dataset['data_loc'][index], dtype=torch.float32) # [max_len_nei + 1, 3]
        else:
            if self.dataset is None:
                self.dataset = h5py.File(self.dataset_path, 'r')
            data = torch.as_tensor(self.dataset['data'][index], dtype=torch.complex64) # [num_ant, num_car]
            loc = self.data_loc[index] # [3]
            if self.use_seq_dataset:
                select = self.exp_conf['select']
                if self.as_scenario_info: # dataset_type = train
                    data, loc = self.find_sequence(data, loc, select, train=True) # [max_len_nei + 1, num_ant, num_car], [max_len_nei + 1, 3]
                else: # dataset_type = test
                    assert self.train_dataset_obj is not None
                    data, loc = self.train_dataset_obj.find_sequence(data, loc, select, train=False) # [max_len_nei + 1, num_ant, num_car], [max_len_nei + 1, 3]
        init_loc = self.init_loc[index]
        return data, loc[..., :2], init_loc[..., :2]
    
    def __len__(self):
        return self.num_data

    def __del__(self):
        if self.dataset is not None:
            self.dataset.close()
        if self.use_seq_dataset and self.gen_seq_datasets:
            if self.seq_dataset is not None:
                self.seq_dataset.close()
    
    def find_sequence(self, data, loc, select, train=False):
        '''
        data: [num_ant, num_car], complex
        loc: [2/3], float
        
        Return
        data_seq: [max_len_nei + 1, num_ant, num_car], complex
        loc_seq: [max_len_nei + 1, 2/3], float
        '''
        assert select in ['random', 'neighbor']
        assert self.as_scenario_info
        if self.dataset is None:
            self.dataset = h5py.File(self.dataset_path, 'r')
        device = data.device
        max_len_nei = self.exp_conf['max_len_nei']
        data_loc = self.data_loc[..., :loc.shape[-1]].to(device) # [num_data, 2/3]
        if select == 'random':
            indices = random.sample(range(self.num_data), max_len_nei)
        elif select == 'neighbor':
            radius = self.exp_conf['radius']
            dist = torch.linalg.norm(loc - data_loc, dim=-1) # [num_data]
            if radius is None:
                indices = torch.argsort(dist).cpu().numpy() # [num_data]
                if train:
                    indices = indices[1:]
                indices = indices[:max_len_nei].tolist()
            else:
                indices = torch.argwhere(torch.logical_and(dist > 0, dist < radius)).squeeze(dim=-1).cpu().numpy()
                if len(indices) >= max_len_nei:
                    np.random.shuffle(indices)
                    indices = indices[:max_len_nei].tolist()
                else: # same as radius is None
                    indices = torch.argsort(dist).cpu().numpy() # [num_data]
                    if train:
                        indices = indices[1:]
                    indices = indices[:max_len_nei].tolist()
        nei_data = []
        nei_loc = []
        for i in indices:
            data_ = torch.as_tensor(self.dataset['data'][i], dtype=torch.complex64, device=device)
            loc_ = data_loc[i]
            nei_data.append(data_)
            nei_loc.append(loc_)
        data_seq = torch.stack(nei_data + [data], dim=0) # [max_len_nei + 1, num_ant, num_car]
        loc_seq = torch.stack(nei_loc + [loc], dim=0) # [max_len_nei + 1, 2/3]
        return data_seq, loc_seq
    
    def generate_sequence_dataset(self):
        assert self.as_scenario_info
        assert self.use_seq_dataset
        assert self.gen_seq_datasets
        if self.dataset is None:
            self.dataset = h5py.File(self.dataset_path, 'r')
        
        num_ant = self.exp_conf['num_ant']
        num_car = self.exp_conf['num_car']
        max_len_nei = self.exp_conf['max_len_nei']
        select = self.exp_conf['select']
        radius = self.exp_conf['radius']
    
        if select == 'neighbor' and radius is not None:
            suffix = f'neighbor_{radius}m'
        else:
            suffix = select
        self.seq_dataset_path = os.path.join(self.data_dir, f'train_{self.num_data // 1000}k_len{max_len_nei}_{suffix}.h5')
        if os.path.exists(self.seq_dataset_path):
            return
        seq_dataset = h5py.File(self.seq_dataset_path, 'w')
        data_dataset = seq_dataset.create_dataset('data',
                                                  (self.num_data, max_len_nei + 1, num_ant, num_car),
                                                  maxshape=(None, max_len_nei + 1, num_ant, num_car),
                                                  chunks=(1, max_len_nei + 1, num_ant, num_car),
                                                  dtype='complex64')
        loc_dataset = seq_dataset.create_dataset('data_loc',
                                                 (self.num_data, max_len_nei + 1, 3),
                                                 maxshape=(None, max_len_nei + 1, 3),
                                                 chunks=(1, max_len_nei + 1, 3),
                                                 dtype='float32')
        for i in tqdm(range(self.num_data), desc=self.seq_dataset_path):
            data = torch.as_tensor(self.dataset['data'][i], dtype=torch.complex64) # [num_ant, num_car]
            loc = self.data_loc[i] # [3]
            data_seq, loc_seq = self.find_sequence(data, loc, select, train=True) # [max_len_nei + 1, num_ant, num_car], [max_len_nei + 1, 3]
            data_dataset[i] = data_seq.cpu().numpy()
            loc_dataset[i] = loc_seq.cpu().numpy()
        seq_dataset.close()
    
    @staticmethod
    def generate_sequence_dataset_test(train_dataset, test_dataset):
        assert train_dataset.as_scenario_info
        assert test_dataset.use_seq_dataset
        assert test_dataset.gen_seq_datasets
        if test_dataset.dataset is None:
            test_dataset.dataset = h5py.File(test_dataset.dataset_path, 'r')
        
        num_ant = test_dataset.exp_conf['num_ant']
        num_car = test_dataset.exp_conf['num_car']
        max_len_nei = test_dataset.exp_conf['max_len_nei']
        select = test_dataset.exp_conf['select']
        radius = test_dataset.exp_conf['radius']
    
        if select == 'neighbor' and radius is not None:
            suffix = f'neighbor_{radius}m'
        else:
            suffix = select
        test_dataset.seq_dataset_path = os.path.join(test_dataset.data_dir, f'test_{train_dataset.num_data // 1000}k_len{max_len_nei}_{suffix}.h5')
        if os.path.exists(test_dataset.seq_dataset_path):
            return
        seq_dataset = h5py.File(test_dataset.seq_dataset_path, 'w')
        data_dataset = seq_dataset.create_dataset('data',
                                                  (test_dataset.num_data, max_len_nei + 1, num_ant, num_car),
                                                  maxshape=(None, max_len_nei + 1, num_ant, num_car),
                                                  chunks=(1, max_len_nei + 1, num_ant, num_car),
                                                  dtype='complex64')
        loc_dataset = seq_dataset.create_dataset('data_loc',
                                                 (test_dataset.num_data, max_len_nei + 1, 3),
                                                 maxshape=(None, max_len_nei + 1, 3),
                                                 chunks=(1, max_len_nei + 1, 3),
                                                 dtype='float32')
        for i in tqdm(range(test_dataset.num_data), desc=test_dataset.seq_dataset_path):
            data = torch.as_tensor(test_dataset.dataset['data'][i], dtype=torch.complex64) # [num_ant, num_car]
            loc = test_dataset.data_loc[i] # [3]
            data_seq, loc_seq = train_dataset.find_sequence(data, loc, select, train=False) # [max_len_nei + 1, num_ant, num_car], [max_len_nei + 1, 3]
            data_dataset[i] = data_seq.cpu().numpy()
            loc_dataset[i] = loc_seq.cpu().numpy()
        seq_dataset.close()