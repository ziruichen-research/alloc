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
test_num=20000

def find_sequence_test(num,data,pos,pos_array,path_now,error_range=1):
    array_num = pos_array.shape[0]
    indices = random.sample(range(array_num), num)
    neighbor_list=[]
    pos_neighbor_list=[]
    for j in indices:
        data_neigh=np.load(path_now+'/data_'+str(j)+'.npy')
        neighbor_list.append(data_neigh)
        pos_neighbor=np.load(path_now+'/data_local_'+str(j)+'.npy')
        pos_neighbor_list.append(pos_neighbor)
    neighbor_list.append(data)
    pos_neighbor_list.append(pos)
    data_sequence=np.array(neighbor_list)
    pos_sequence=np.array(pos_neighbor_list)
    return data_sequence,pos_sequence


path_list=['/home/chenzirui/Desktop/home_mnt/multiplescenario_bothside/32ant*32car_3.5GHz_40MHz_BS2_R101-700_V1_24.5.26/train20000','/home/chenzirui/Desktop/home_mnt/multiplescenario_bothside/32ant*32car_3.5GHz_40MHz_BS3_R701-1200_V1_24.5.26/train20000',
           '/home/chenzirui/Desktop/home_mnt/multiplescenario_bothside/32ant*32car_3.5GHz_40MHz_BS5_R1201-1600_V1_24.5.26/train20000','/home/chenzirui/Desktop/home_mnt/multiplescenario_bothside/32ant*32car_3.5GHz_40MHz_BS8_R1601-2000_V1_24.5.26/train20000',
           '/home/chenzirui/Desktop/home_mnt/multiplescenario_bothside/32ant*32car_3.5GHz_40MHz_BS9_R2001-2700_V1_24.5.26/train20000']
for kk,path in enumerate(path_list):
    pos_array=np.load(path+'/train'+'/pos_array.npy')
    os.makedirs(path + '/sequence_test_len64_notneigh', exist_ok=True)
    path_now=path+'/train'
    test_num=20000
    for i in tqdm(range(test_num)):
        test_data = np.load(path + '/test' + '/data_' + str(i) + '.npy')
        test_local = np.load(path + '/test' + '/data_local_' + str(i) + '.npy')
        data_sequence, pos_sequence = find_sequence_test(num=64, data=test_data, pos=test_local, pos_array=pos_array,
                                                         path_now=path + '/train')
        np.save(path + '/sequence_test_len64_notneigh' + '/data_' + str(i) + '.npy', data_sequence)
        np.save(path + '/sequence_test_len64_notneigh' + '/data_local_' + str(i) + '.npy', pos_sequence)
    np.save(path + '/sequence_test_len64_notneigh' + '/data_len.npy', test_num)
