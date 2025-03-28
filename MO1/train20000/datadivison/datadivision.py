import numpy as np
import scipy.io as scio
import hdf5storage
import random
import os
gpu_list = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
from tqdm import tqdm
gpu_list = '0'

np.random.seed(1)
random.seed(1)

path_list=['/home/chenzirui/Desktop/home_mnt/multiplescenario_bothside/32ant*32car_3.5GHz_40MHz_BS2_R101-700_V1_24.5.26','/home/chenzirui/Desktop/home_mnt/multiplescenario_bothside/32ant*32car_3.5GHz_40MHz_BS3_R701-1200_V1_24.5.26',
           '/home/chenzirui/Desktop/home_mnt/multiplescenario_bothside/32ant*32car_3.5GHz_40MHz_BS5_R1201-1600_V1_24.5.26','/home/chenzirui/Desktop/home_mnt/multiplescenario_bothside/32ant*32car_3.5GHz_40MHz_BS8_R1601-2000_V1_24.5.26',
           '/home/chenzirui/Desktop/home_mnt/multiplescenario_bothside/32ant*32car_3.5GHz_40MHz_BS9_R2001-2700_V1_24.5.26']
num_list=[181*600,181*500,181*400,181*400,181*700]
BS_local=[[287.5040,389.5040,6],[235.5040,489.5040,6],[235.5040,589.5040,6],[287.5040,651.5040,6],[235.5040,751.5040,6]]

for kk,path in enumerate(path_list):
    data_mat_local=hdf5storage.loadmat(path+'/mat'+'/DeepMIMO_local.mat')
    data_local=data_mat_local['DeepMIMO_loc']
    print(data_local.shape)


    os.makedirs(path+'/train20000',exist_ok=True)
    os.makedirs(path+'/train20000/train',exist_ok=True)
    os.makedirs(path+'/train20000/test',exist_ok=True)

    index=np.arange(num_list[kk])
    np.random.shuffle(index)
    train_num=20000
    test_num=20000
    data_local=data_local-BS_local[kk]


    pos_array = data_local[index[:train_num]]
    print(pos_array.shape)
    np.save(path+'/train20000/train'+'/pos_array.npy',pos_array)
    print("pos_array",pos_array)

    for i in tqdm(range(train_num)):
        data_mat = hdf5storage.loadmat(path + '/mat' + '/DeepMIMO_dataset_' + str(index[i]) + '.mat')
        data = data_mat['DeepMIMO_dataset']
        np.save(path + '/train20000/train' + '/data_' + str(i) + '.npy', data)
        np.save(path + '/train20000/train' + '/data_local_' + str(i) + '.npy', data_local[index[i]])
    np.save(path+'/train20000/train'+'/data_len.npy',train_num)

    for i in tqdm(range(test_num)):
        data_mat = hdf5storage.loadmat(path + '/mat' + '/DeepMIMO_dataset_' + str(index[i+20000]) + '.mat')
        data = data_mat['DeepMIMO_dataset']
        np.save(path + '/train20000/test' + '/data_' + str(i) + '.npy', data)
        np.save(path + '/train20000/test' + '/data_local_' + str(i) + '.npy', data_local[index[i+20000]])
    np.save(path+'/train20000/test'+'/data_len.npy',test_num)


