import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
import hdf5storage
import random
import os
gpu_list = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
from tqdm import tqdm



path_list=['/home/chenzirui/Desktop/home_mnt/multiplescenario_bothside/32ant*32car_3.5GHz_40MHz_BS2_R101-700_V1_24.5.26','/home/chenzirui/Desktop/home_mnt/multiplescenario_bothside/32ant*32car_3.5GHz_40MHz_BS3_R701-1200_V1_24.5.26',
           '/home/chenzirui/Desktop/home_mnt/multiplescenario_bothside/32ant*32car_3.5GHz_40MHz_BS5_R1201-1600_V1_24.5.26','/home/chenzirui/Desktop/home_mnt/multiplescenario_bothside/32ant*32car_3.5GHz_40MHz_BS8_R1601-2000_V1_24.5.26',
           '/home/chenzirui/Desktop/home_mnt/multiplescenario_bothside/32ant*32car_3.5GHz_40MHz_BS9_R2001-2700_V1_24.5.26']
num_list=[181*600,181*500,181*400,181*400,181*700]
BS_local=[[287.5040,389.5040,6],[235.5040,489.5040,6],[235.5040,589.5040,6],[287.5040,651.5040,6],[235.5040,751.5040,6]]

x_list=list(range(2250,450,-1))
y_list=[]


num_points=len(x_list)

jiedian_list=[(2250,80),(1950,80),(1850,40),(1550,40),(1450,80),(1150,80),(1050,120),(750,120),(650,80),(450,80)]


for kk,jiedian in enumerate(jiedian_list):
    if kk >0:
        y_list.extend(np.linspace(jiedian_list[kk-1][1],jiedian_list[kk][1],jiedian_list[kk-1][0]-jiedian_list[kk][0]))

y_list = np.round(y_list).astype(int)



def get_BS_index(x_index):
    thresholds = [2001, 1601, 1201, 701, 101]
    for i, threshold in enumerate(thresholds):
        if x_index >= threshold:
            return len(thresholds) - i-1

def get_channel_index(x_index,y_index):
    thresholds = [2001, 1601, 1201, 701, 101]
    for i, threshold in enumerate(thresholds):
        if x_index >= threshold:
            new_x_index=x_index-threshold
            break
    sum_index=int(new_x_index*181+y_index)
    return sum_index

def get_true_local(channel_index,BS_index):
    now_path=path_list[BS_index]
    data_mat_local = hdf5storage.loadmat(now_path + '/mat' + '/DeepMIMO_local.mat')
    data_local = data_mat_local['DeepMIMO_loc']
    now_local=data_local[channel_index]
    return now_local

def add_noise_loc(true_local,error_range=2):
    error = np.zeros_like(true_local)
    error[0:2] = np.random.uniform(-1 * error_range, error_range, size=2)
    initial_local = true_local + error
    return initial_local


np.random.seed(0)
random.seed(5)
error_range_list=[0.5,5]
for error_range in error_range_list:
    BS_index_list=[]
    channel_index_list=[]
    true_local_list=[]
    initial_local_list=[]
    for i in tqdm(range(len(x_list))):
        BS_index_list.append(get_BS_index(x_list[i]))
        channel_index_list.append(get_channel_index(x_list[i],y_list[i]))
        true_local_list.append(get_true_local(channel_index_list[i],BS_index_list[i]))
        initial_local_list.append(add_noise_loc(true_local_list[i],error_range=error_range))




    data_dict = {
        'BS_index_list': BS_index_list,
        'channel_index_list': channel_index_list,
        'true_local_list': true_local_list,
        'initial_local_list': initial_local_list
    }



    array = np.array([data_dict], dtype=object)
    np.save('trace_information_errorrange'+str(error_range)+'.npy',array)


from matplotlib import rcParams
