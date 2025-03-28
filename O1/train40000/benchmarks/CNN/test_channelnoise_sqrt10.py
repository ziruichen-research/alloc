import os
gpu_list = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
import numpy as np
import math
import torch
import torch.nn as nn
import random
from tqdm import tqdm
from model import *
from thop import profile

cpu_num = 4
os.environ["OMP_NUM_THREADS"] = str(cpu_num)
os.environ["MKL_NUM_THREADS"] = str(cpu_num)
torch.set_num_threads(cpu_num )
gpu_list = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list

batch_size = 500
hidden_size=32

car_size=32
ant_size=32

path='/home/chenzirui/Desktop/home_mnt/32ant*32car_3.5GHz_40MHz_R501-1400_V1_24.4.12/train40000'
train_dataset = DatasetFolder_mapping(path+'/train')
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,pin_memory=True, drop_last=True)

test_dataset=DatasetFolder_mapping(path+'/test')
test_loader=torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True, num_workers=4,pin_memory=True, drop_last=True)

model=LocNet(num_ant=ant_size,num_car=car_size,hidden_size=hidden_size)

if len(gpu_list.split(',')) > 1:
    model = torch.nn.DataParallel(model).cuda()
else:
    model = model.cuda()
model.load_state_dict(torch.load('./model' +'.pth'),strict=False)

model.eval()
noise_sig_list=[0,0.01,10**-1.5,0.1,10**-0.5,1]
avg_rmse_list=[]

model.eval()
for noise_sig in noise_sig_list:
    sum_rmse = 0
    for i, (data, local) in tqdm(enumerate(test_loader),total=len(test_loader)):
        data = data.cuda().float()
        data = data * 10000
        local = local.cuda().float()
        local_current = local[:, 0:2]
        noise = noise_sig * torch.randn_like(data)
        noise = noise.cuda()
        noise = 1 + noise
        data = data * noise
        with torch.no_grad():
            local_pred = model(data)
        rmse = RMSE(local_pred, local_current)
        rmse=rmse.item()
        sum_rmse+=rmse
    avg_rmse=sum_rmse/(i+1)
    avg_rmse_list.append(avg_rmse)
print("avg_rmse_list:", avg_rmse_list)
