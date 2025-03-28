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

path_list=['/home/chenzirui/Desktop/home_mnt/multiplescenario_bothside/32ant*32car_3.5GHz_40MHz_BS2_R101-700_V1_24.5.26/train20000','/home/chenzirui/Desktop/home_mnt/multiplescenario_bothside/32ant*32car_3.5GHz_40MHz_BS3_R701-1200_V1_24.5.26/train20000',
           '/home/chenzirui/Desktop/home_mnt/multiplescenario_bothside/32ant*32car_3.5GHz_40MHz_BS5_R1201-1600_V1_24.5.26/train20000','/home/chenzirui/Desktop/home_mnt/multiplescenario_bothside/32ant*32car_3.5GHz_40MHz_BS8_R1601-2000_V1_24.5.26/train20000',
           '/home/chenzirui/Desktop/home_mnt/multiplescenario_bothside/32ant*32car_3.5GHz_40MHz_BS9_R2001-2700_V1_24.5.26/train20000']

for kk,path in enumerate(path_list):
    os.makedirs(path+'/sequence',exist_ok=True)
    train_num=20000

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