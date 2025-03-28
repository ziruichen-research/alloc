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

batch_size = 1
hidden_size=128

car_size=32
ant_size=32

path='/home/chenzirui/Desktop/home_mnt/32ant*32car_3.5GHz_40MHz_O1B_R501-1400_V1_24.4.12/train20000'
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
rmse_distribution_list=[]
sum_rmse = 0
model.eval()
for i, (data, local) in tqdm(enumerate(test_loader),total=len(test_loader)):
    data = data.cuda().float()
    data = data * 10000
    local = local.cuda().float()
    local_current = local[:, 0:2]
    with torch.no_grad():
        local_pred = model(data)
    local_pred=local_pred[:,-1,:]
    rmse = RMSE(local_pred, local_current)
    rmse=rmse.item()
    rmse_distribution_list.append(rmse)
    sum_rmse+=rmse
avg_rmse=sum_rmse/(i+1)
print("avg_distance :", avg_rmse)

rmse_distribution_list.sort()
total_len=len(rmse_distribution_list)
rate=0.2
print("rate:",rate)
low_len=int(rate*total_len)
print("low :",rmse_distribution_list[low_len])
high_len=int((1-rate)*total_len)
print("high :",rmse_distribution_list[high_len])
rate=0.1
print("rate:",rate)
low_len=int(rate*total_len)
print("low :",rmse_distribution_list[low_len])
high_len=int((1-rate)*total_len)
print("high :",rmse_distribution_list[high_len])

rmse_distribution_array=np.array(rmse_distribution_list)
np.save("rmse_distribution.npy",rmse_distribution_array)