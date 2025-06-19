import numpy as np
import scipy.io as scio
import hdf5storage
import random
import os
gpu_list = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
from tqdm import tqdm
from einops import rearrange,repeat
gpu_list = '0'

np.random.seed(1)
random.seed(1)

def find_sequence(num, data, pos, pos_array,
                  path_now):

    current_pos = repeat(pos, 'l -> train_num l', train_num=pos_array.shape[0])
    distance_array = np.linalg.norm(current_pos - pos_array, axis=1)
    indices = (np.argpartition(distance_array, num + 1)[:num + 1]).tolist()
    indices.remove((np.argpartition(distance_array, 1)[0]).item())
    neighbor_list = []
    pos_neighbor_list = []
    for j in indices:
        data_neigh = np.load(path_now + '/data_' + str(j) + '.npy')
        neighbor_list.append(data_neigh)
        pos_neighbor = np.load(path_now + '/data_local_' + str(j) + '.npy')
        pos_neighbor_list.append(pos_neighbor)
    neighbor_list.append(data)
    pos_neighbor_list.append(pos)
    data_sequence = np.array(neighbor_list)
    pos_sequence = np.array(pos_neighbor_list)
    return data_sequence, pos_sequence

path_total='/home/chenzirui/Desktop/home_mnt/rainscenario_32ant*32car_60GHz_1600MHz_BS3_V2_25.6.1'
path_scenario_index=['/rainfall0','/rainfall10','/rainfall20','/rainfall30','/rainfall40','/rainfall50','/rainfall60','/rainfall70','/rainfall80','/rainfall90']
path_list=[path_total+scenario_index+'/train40000' for scenario_index in path_scenario_index]

for kk,path in enumerate(path_list):
    os.makedirs(path+'/sequence',exist_ok=True)
    train_num=40000

    num=64
    pos_array=np.load(path+'/train'+'/pos_array.npy')
    path_now=path+'/train'
    for i in tqdm(range(train_num)):
        data=np.load(path + '/train' + '/data_' + str(i) + '.npy')
        pos_error=pos_array[i]
        data_sequence,pos_sequence=find_sequence(num=num,data=data,pos=pos_error,pos_array=pos_array,path_now=path_now)
        np.save(path+'/sequence'+'/data_'+ str(i) + '.npy',data_sequence)
        np.save(path + '/sequence' + '/data_local_' + str(i) + '.npy', pos_sequence)
    np.save(path+'/sequence'+'/data_len.npy',train_num)