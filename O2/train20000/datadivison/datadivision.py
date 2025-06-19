import numpy as np
import scipy.io as scio
import hdf5storage
import random
import os
from pathlib import Path

gpu_list = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
from tqdm import tqdm

np.random.seed(1)
random.seed(1)

path_total='/home/chenzirui/Desktop/home_mnt/dynamicscenario_50scenarios_32ant_32car_40MHz_BS1_25.5.31'
path_scenario_index=[f'/scenario{6 + i * 20}' for i in range(50)]
print(path_scenario_index)
path_list=[path_total+scenario_index for scenario_index in path_scenario_index]
BS_local=[3,10,6]
num_list=[]
min_num=40000
for kk,path in enumerate(path_list):
    folder_path = Path(path + '/mat')
    num = sum(1 for item in folder_path.iterdir() if item.is_file()) - 1
    if num < min_num:
        min_num=num
print("min_num",min_num)
if min_num <= 40000-1:
    exit()


for kk,path in enumerate(path_list):
    data_mat_local=hdf5storage.loadmat(path+'/mat'+'/DeepMIMO_local.mat')
    data_local=data_mat_local['DeepMIMO_loc']
    print(data_local.shape)


    os.makedirs(path+'/train20000',exist_ok=True)
    os.makedirs(path+'/train20000/train',exist_ok=True)
    os.makedirs(path+'/train20000/test',exist_ok=True)
    folder_path=Path(path+'/mat')
    num=sum(1 for item in folder_path.iterdir() if item.is_file())-1
    print("num",num)
    num_list.append(num)
    index=np.arange(num)
    np.random.shuffle(index)
    train_num=20000
    test_num=20000
    data_local=data_local-BS_local


    pos_array = data_local[index[:train_num]]
    print(pos_array.shape)
    np.save(path+'/train20000/train'+'/pos_array.npy',pos_array)
    print("pos_array",pos_array)

    for i in tqdm(range(train_num)):
        data_mat = hdf5storage.loadmat(path + '/mat' + '/DeepMIMO_dataset_' + str(index[i]) + '.mat')
        data = data_mat['mat_m']
        np.save(path + '/train20000/train' + '/data_' + str(i) + '.npy', data)
        np.save(path + '/train20000/train' + '/data_local_' + str(i) + '.npy', data_local[index[i]])
    np.save(path+'/train20000/train'+'/data_len.npy',train_num)

    for i in tqdm(range(test_num)):
        data_mat = hdf5storage.loadmat(path + '/mat' + '/DeepMIMO_dataset_' + str(index[i+20000]) + '.mat')
        data = data_mat['mat_m']
        np.save(path + '/train20000/test' + '/data_' + str(i) + '.npy', data)
        np.save(path + '/train20000/test' + '/data_local_' + str(i) + '.npy', data_local[index[i+20000]])
    np.save(path+'/train20000/test'+'/data_len.npy',test_num)

print("num_list:",num_list)
