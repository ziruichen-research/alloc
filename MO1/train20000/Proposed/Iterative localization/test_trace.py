import os
gpu_list = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
import numpy as np
import math
import torch
import torch.nn as nn
import random
from model import *
from thop import profile
import time
from ptflops import get_model_complexity_info
from torchstat import stat
from einops import rearrange,repeat
from tqdm import tqdm
import hdf5storage

cpu_num = 4
os.environ["OMP_NUM_THREADS"] = str(cpu_num)
os.environ["MKL_NUM_THREADS"] = str(cpu_num)
torch.set_num_threads(cpu_num )

np.random.seed(1)
random.seed(1)

batch_size=1
ant_size=32
car_size=32
hidden_size=256

depth=8

model=Nlocnet(ant_size=ant_size,car_size=car_size,embed_dim=hidden_size,num_heads=4,hidden_dim=hidden_size,depth=depth)

if len(gpu_list.split(',')) > 1:
    model = torch.nn.DataParallel(model).cuda()
else:
    model = model.cuda()

path_list=['/home/chenzirui/Desktop/home_mnt/multiplescenario_bothside/32ant*32car_3.5GHz_40MHz_BS2_R101-700_V1_24.5.26','/home/chenzirui/Desktop/home_mnt/multiplescenario_bothside/32ant*32car_3.5GHz_40MHz_BS3_R701-1200_V1_24.5.26',
           '/home/chenzirui/Desktop/home_mnt/multiplescenario_bothside/32ant*32car_3.5GHz_40MHz_BS5_R1201-1600_V1_24.5.26','/home/chenzirui/Desktop/home_mnt/multiplescenario_bothside/32ant*32car_3.5GHz_40MHz_BS8_R1601-2000_V1_24.5.26',
           '/home/chenzirui/Desktop/home_mnt/multiplescenario_bothside/32ant*32car_3.5GHz_40MHz_BS9_R2001-2700_V1_24.5.26']

num_list=[181*600,181*500,181*400,181*400,181*700]
BS_local=[[287.5040,389.5040,6],[235.5040,489.5040,6],[235.5040,589.5040,6],[287.5040,651.5040,6],[235.5040,751.5040,6]]

def find_sequence_test_multiscenario(num,channel_index,BS_index,initial_local,path_list):

    path_now=path_list[BS_index]
    pos_array=np.load(path_now+'/train20000/train' + '/pos_array.npy')
    data=hdf5storage.loadmat(path_list[BS_index]+'/mat' + '/DeepMIMO_dataset_' + str(channel_index) + '.mat')['DeepMIMO_dataset']
    initial_local=initial_local-BS_local[BS_index]
    current_pos=repeat(initial_local,'l -> train_num l',train_num=pos_array.shape[0])
    distance_array=np.linalg.norm(current_pos-pos_array,axis=1)
    indices = (np.argpartition(distance_array, num)[:num]).tolist()
    neighbor_list=[]
    pos_neighbor_list=[]
    for j in indices:
        data_neigh=np.load(path_now+'/train20000/train'+'/data_'+str(j)+'.npy')
        neighbor_list.append(data_neigh)
        pos_neighbor=np.load(path_now+'/train20000/train'+'/data_local_'+str(j)+'.npy')
        pos_neighbor_list.append(pos_neighbor)
    neighbor_list.append(data)
    pos_neighbor_list.append(initial_local)
    data_sequence=np.array(neighbor_list)
    pos_sequence=np.array(pos_neighbor_list)
    return data_sequence,pos_sequence

def find_sequence_test_multiscenario_random(num,channel_index,BS_index,path_list):

    path_now=path_list[BS_index]
    pos_array=np.load(path_now+'/train20000/train' + '/pos_array.npy')
    data=hdf5storage.loadmat(path_list[BS_index]+'/mat' + '/DeepMIMO_dataset_' + str(channel_index) + '.mat')['DeepMIMO_dataset']
    array_num = pos_array.shape[0]
    indices = random.sample(range(array_num), num)
    neighbor_list=[]
    pos_neighbor_list=[]
    for j in indices:
        data_neigh=np.load(path_now+'/train20000/train'+'/data_'+str(j)+'.npy')
        neighbor_list.append(data_neigh)
        pos_neighbor=np.load(path_now+'/train20000/train'+'/data_local_'+str(j)+'.npy')
        pos_neighbor_list.append(pos_neighbor)
    neighbor_list.append(data)
    pos_neighbor_list.append(pos_neighbor)
    data_sequence=np.array(neighbor_list)
    pos_sequence=np.array(pos_neighbor_list)
    return data_sequence,pos_sequence

error_range_list=[0]
for error_range in error_range_list:

    trace_information_array=np.load('trace_information_errorrange'+str(error_range)+'.npy',allow_pickle=True)
    BS_index_list=trace_information_array[0]['BS_index_list']
    channel_index_list=trace_information_array[0]['channel_index_list']
    true_local_list=trace_information_array[0]['true_local_list']

    neigh_num=64
    inferred_local_list_step1 = []
    error_list_step1 = []
    path_model_notneigh = '/home/chenzirui/Desktop/ALLoc/Multiplescenario_bothside/train20000/Proposed/Scheme_notneigh/'
    model.load_state_dict(torch.load(path_model_notneigh + './model' + '.pth'), strict=False)
    for i in tqdm(range(len(BS_index_list))):
        BS_index=BS_index_list[i]
        channel_index=channel_index_list[i]
        true_local=true_local_list[i]
        data_sequence,pos_sequence=find_sequence_test_multiscenario_random(neigh_num,channel_index,BS_index,path_list)

        data = torch.tensor(data_sequence).unsqueeze(0)
        local = torch.tensor(pos_sequence).unsqueeze(0)
        data = data.cuda().float()
        local = local.cuda().float()
        local = local[:, :, 0:2]
        data = data * 10000
        data_neigh = data[:, 0:-1, ...]
        data_current = data[:, -1:, ...]
        local_neigh = local[:, 0:-1, ...]
        local_current = local[:, -1:, ...]
        with torch.no_grad():
            local_pred = model(data_neigh, local_neigh, data_current)
        inferred_local=local_pred[0,-1,:].cpu().numpy()
        inferred_local=inferred_local+BS_local[BS_index][0:2]

        inferred_local_list_step1.append(inferred_local)
        error=np.linalg.norm(inferred_local-true_local[0:2])
        error_list_step1.append(error)
    print('step2 avg_error',sum(error_list_step1)/len(error_list_step1))

    inferred_local_step1 = np.array(inferred_local_list_step1)
    np.save('inferred_local_step1'+'.npy',inferred_local_step1)
    print(len(inferred_local_list_step1))

    inferred_local_list_step2 = []
    error_list_step2 = []
    path_model = '/home/chenzirui/Desktop/ALLoc/Multiplescenario_bothside/train20000/Proposed/Scheme/'
    model.load_state_dict(torch.load(path_model + './model' + '.pth'), strict=False)
    for i in tqdm(range(len(BS_index_list))):
        BS_index = BS_index_list[i]
        channel_index = channel_index_list[i]
        true_local = true_local_list[i]
        initial_local= inferred_local_list_step1[i]
        threesize_initial_local=np.zeros_like(true_local)
        threesize_initial_local[0:2]=initial_local
        threesize_initial_local[2]=true_local[2]
        initial_local=threesize_initial_local
        data_sequence, pos_sequence = find_sequence_test_multiscenario(neigh_num, channel_index, BS_index,initial_local, path_list)

        data = torch.tensor(data_sequence).unsqueeze(0)
        local = torch.tensor(pos_sequence).unsqueeze(0)
        data = data.cuda().float()
        local = local.cuda().float()
        local = local[:, :, 0:2]
        data = data * 10000
        data_neigh = data[:, 0:-1, ...]
        data_current = data[:, -1:, ...]
        local_neigh = local[:, 0:-1, ...]
        local_current = local[:, -1:, ...]
        with torch.no_grad():
            local_pred = model(data_neigh, local_neigh, data_current)
        inferred_local = local_pred[0, -1, :].cpu().numpy()
        inferred_local = inferred_local + BS_local[BS_index][0:2]

        inferred_local_list_step2.append(inferred_local)
        error = np.linalg.norm(inferred_local - true_local[0:2])
        error_list_step2.append(error)
    print('step2 avg_error', sum(error_list_step2) / len(error_list_step2))

    inferred_local_step2 = np.array(inferred_local_list_step2)
    np.save('inferred_local_step2' + '.npy', inferred_local_step2)
