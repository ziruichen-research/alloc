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

cpu_num = 4
os.environ["OMP_NUM_THREADS"] = str(cpu_num)
os.environ["MKL_NUM_THREADS"] = str(cpu_num)
torch.set_num_threads(cpu_num )

gpu_list = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list

np.random.seed(1)
random.seed(1)

batch_size=1
epochs=100000000
steps=300000
learning_rate=1e-4
print_freq=200

ant_size=32
car_size=32
hidden_size=256

depth=8

model=Nlocnet(ant_size=ant_size,car_size=car_size,embed_dim=hidden_size,num_heads=4,hidden_dim=hidden_size,depth=depth)


if len(gpu_list.split(',')) > 1:
    model = torch.nn.DataParallel(model).cuda()
else:
    model = model.cuda()

model.load_state_dict(torch.load('./model' +'.pth'),strict=False)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print('#model parameters:', sum(param.numel() for param in model.parameters()))

path_total='/home/chenzirui/Desktop/home_mnt/dynamicscenario_50scenarios_32ant_32car_40MHz_BS1_25.5.31'
path_scenario_index=[f'/scenario{6 + i * 20}' for i in range(50)]
path_list=[path_total+scenario_index+'/train20000' for scenario_index in path_scenario_index]
train_loader_list=[]
test_loader_list=[]
for path in path_list:
    train_dataset = DatasetFolder(path + '/sequence')
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    test_dataset = DatasetFolder(path + '/sequence_test_len64')
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    train_loader_list.append(train_loader)
    test_loader_list.append(test_loader)

model.eval()
test_scenario_index=[item*5+1 for item in [1,2,5,7,8]]
evaluated_scenario_index = test_scenario_index + [6*5+1]
for kk, test_loader in enumerate(test_loader_list):
    if kk in evaluated_scenario_index:
        rmse_distribution_list = []
        sum_rmse = 0
        for i, (data, local) in tqdm(enumerate(test_loader),total=len(test_loader)):
            data = data.cuda().float()
            data = data * 100000
            local = local.cuda().float()
            local = local[:, :, 0:2]
            data_neigh = data[:, 0:-1, ...]
            data_current = data[:, -1:, ...]
            local_neigh = local[:, 0:-1, ...]
            local_current = local[:, -1:, ...]
            with torch.no_grad():
                local_pred = model(data_neigh, local_neigh, data_current)
            rmse = RMSE(local_pred[:,-1,:], local_current[:,-1,:])
            rmse = rmse.item()
            rmse_distribution_list.append(rmse)
            sum_rmse+=rmse
        avg_rmse=sum_rmse/(i+1)
        print("scenario index:",(kk-1)//5*100+26)
        print("avg_distance :",avg_rmse)

        rmse_distribution_list.sort()
        total_len = len(rmse_distribution_list)
        rate = 0.2
        print("rate:", rate)
        low_len = int(rate * total_len)
        print("low :", rmse_distribution_list[low_len])
        high_len = int((1 - rate) * total_len)
        print("high :", rmse_distribution_list[high_len])
        rate = 0.1
        print("rate:", rate)
        low_len = int(rate * total_len)
        print("low :", rmse_distribution_list[low_len])
        high_len = int((1 - rate) * total_len)
        print("high :", rmse_distribution_list[high_len])