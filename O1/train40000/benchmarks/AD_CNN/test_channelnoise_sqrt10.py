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

def DFT_matrix_tensor(num_ant,num_car):

    F_delay = torch.zeros([num_car, num_car], dtype=torch.cfloat)
    F_angle = torch.zeros([num_ant, num_ant], dtype=torch.cfloat)
    for i in range(num_car):
        for j in range(num_car):
            F_delay[i, j] = torch.tensor(1 / np.sqrt(num_car) * np.exp(-1j * 2 * np.pi / num_car * i * j))

    for i in range(num_ant):
        for j in range(num_ant):
            F_angle[i, j] = torch.tensor(1 / np.sqrt(num_ant) * np.exp(-1j * 2 * np.pi / num_ant * i * (j - num_ant / 2)))

    return F_delay,F_angle

def DFT_tensor(num_ant,num_car,data,F_delay,F_angle):

    num=data.shape[0]
    data=data.reshape([num,num_ant,num_car,2])
    data_real=data[:,:,:,0]
    data_imag=data[:,:,:,1]
    data_complex=data_real+1j*data_imag

    F_delay_batch=(F_delay.unsqueeze(0)).repeat(num,1,1)
    F_angle_batch=(F_angle.unsqueeze(0)).repeat(num,1,1)
    data_DFTmatrix_complex = torch.bmm(torch.bmm(F_angle_batch,data_complex),(F_delay_batch.real-1j*F_delay_batch.imag).transpose(1,2))
    data_DFTmatrix_real=data_DFTmatrix_complex.real
    data_DFTmatrix_imag=data_DFTmatrix_complex.imag
    data_DFT=torch.zeros([num,num_ant,num_car,2])
    data_DFT[:,:,:,0]=data_DFTmatrix_real
    data_DFT[:,:,:,1]=data_DFTmatrix_imag
    return data_DFT


def DFT_tensor_reverse(num_ant, num_car, data_reverse, F_delay, F_angle):
    num = data_reverse.shape[0]
    data_reverse = data_reverse.reshape([num, num_ant, num_car, 2])
    data_reverse_real = data_reverse[:, :, :, 0]
    data_reverse_imag = data_reverse[:, :, :, 1]
    F_delay_inv = torch.linalg.inv(F_delay)
    F_angle_inv = torch.linalg.inv(F_angle)
    F_delay_inv_batch = (F_delay_inv.unsqueeze(0)).repeat(num, 1, 1)
    F_angle_inv_batch = (F_angle_inv.unsqueeze(0)).repeat(num, 1, 1)
    data_reverse_complex = data_reverse_real + 1j * data_reverse_imag

    data_complex = torch.bmm(torch.bmm(F_angle_inv_batch, data_reverse_complex),
                             ((F_delay_inv_batch.real - 1j * F_delay_inv_batch.imag).transpose(1, 2)))
    data_real = data_complex.real
    data_imag = data_complex.imag
    data = torch.zeros([num, num_ant, num_car, 2])
    data[:, :, :, 0] = data_real
    data[:, :, :, 1] = data_imag
    return data

F_delay,F_angle=DFT_matrix_tensor(num_ant=ant_size,num_car=car_size)

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
        data = data * 10000
        data = DFT_tensor(num_ant=ant_size, num_car=car_size, data=data.float(), F_delay=F_delay,
                          F_angle=F_angle)
        data = data.cuda()
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
