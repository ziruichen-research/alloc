import numpy as np
import scipy.io as scio
import hdf5storage
import random
import os
gpu_list = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
from tqdm import tqdm
gpu_list = '0'

np.random.seed(0)
random.seed(0)

path_total='/home/chenzirui/Desktop/home_mnt/rainscenario_32ant*32car_60GHz_1600MHz_BS3_V2_25.6.1'
path_scenario_index=['/rainfall0','/rainfall10','/rainfall20','/rainfall30','/rainfall40',
                     '/rainfall50','/rainfall60','/rainfall70','/rainfall80','/rainfall90']
path_list=[path_total+scenario_index for scenario_index in path_scenario_index]

num=162900
index=np.arange(num)
np.random.shuffle(index)
print(index)
for kk,path in enumerate(path_list):
    data_mat_local=hdf5storage.loadmat(path+'/mat'+'/DeepMIMO_local.mat')
    data_local=data_mat_local['DeepMIMO_loc']
    print(data_local.shape)


    os.makedirs(path+'/train40000',exist_ok=True)
    os.makedirs(path+'/train40000/train',exist_ok=True)
    os.makedirs(path+'/train40000/test',exist_ok=True)

    train_num=40000
    test_num=20000


    pos_array = data_local[index[:train_num]]
    print(pos_array.shape)
    np.save(path+'/train40000/train'+'/pos_array.npy',pos_array)
    print("pos_array",pos_array)

    for i in tqdm(range(train_num)):
        data_mat = hdf5storage.loadmat(path + '/mat' + '/DeepMIMO_dataset_' + str(index[i]) + '.mat')
        data = data_mat['mat_m']
        np.save(path + '/train40000/train' + '/data_' + str(i) + '.npy', data)
        np.save(path + '/train40000/train' + '/data_local_' + str(i) + '.npy', data_local[index[i]])
    np.save(path+'/train40000/train'+'/data_len.npy',train_num)

    for i in tqdm(range(test_num)):
        data_mat = hdf5storage.loadmat(path + '/mat' + '/DeepMIMO_dataset_' + str(index[i+80000]) + '.mat')
        data = data_mat['mat_m']
        np.save(path + '/train40000/test' + '/data_' + str(i) + '.npy', data)
        np.save(path + '/train40000/test' + '/data_local_' + str(i) + '.npy', data_local[index[i+80000]])
    np.save(path+'/train40000/test'+'/data_len.npy',test_num)


