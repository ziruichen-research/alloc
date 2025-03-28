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

cpu_num = 4
os.environ["OMP_NUM_THREADS"] = str(cpu_num)
os.environ["MKL_NUM_THREADS"] = str(cpu_num)
torch.set_num_threads(cpu_num )


batch_size = 500
epochs = 1000000
steps=300000
learning_rate = 1e-3
print_freq = 200
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

path_list=['/home/chenzirui/Desktop/home_mnt/multiplescenario_bothside/32ant*32car_3.5GHz_40MHz_BS2_R101-700_V1_24.5.26/train20000','/home/chenzirui/Desktop/home_mnt/multiplescenario_bothside/32ant*32car_3.5GHz_40MHz_BS3_R701-1200_V1_24.5.26/train20000',
           '/home/chenzirui/Desktop/home_mnt/multiplescenario_bothside/32ant*32car_3.5GHz_40MHz_BS5_R1201-1600_V1_24.5.26/train20000','/home/chenzirui/Desktop/home_mnt/multiplescenario_bothside/32ant*32car_3.5GHz_40MHz_BS8_R1601-2000_V1_24.5.26/train20000',
           '/home/chenzirui/Desktop/home_mnt/multiplescenario_bothside/32ant*32car_3.5GHz_40MHz_BS9_R2001-2700_V1_24.5.26/train20000']
train_loader_list=[]
test_loader_list=[]
for path in path_list:
    train_dataset = DatasetFolder_mapping(path + '/train')
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    test_dataset = DatasetFolder_mapping(path + '/test')
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    train_loader_list.append(train_loader)
    test_loader_list.append(test_loader)

num_datasets=5
model=LocNet(num_ant=ant_size,num_car=car_size,hidden_size=hidden_size,num_datasets=num_datasets)
from calflops import calculate_flops
input1=torch.randn(1,ant_size,car_size,2)
idx=1
inputs={
    "data":input1,
}
flops, macs, params = calculate_flops(model=model, kwargs=inputs, print_results=True)
print("FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))

if len(gpu_list.split(',')) > 1:
    model = torch.nn.DataParallel(model).cuda()
else:
    model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

class LrSchedule():
    def __init__(self,initial_lr,optimizer=None):
        super(LrSchedule,self).__init__()
        self.steps=0
        self.initial_lr=initial_lr
        self.optimizer=optimizer

    def step(self):
        self.steps+=1
        if self.steps <= 150000 :
            lr=self.initial_lr
        elif self.steps > 150000 and self.steps <= 200000:
            lr=self.initial_lr * (0.2 ** 1)
        elif self.steps > 200000 and self.steps <= 250000:
            lr=self.initial_lr * (0.2 ** 2)
        elif self.steps > 250000 and self.steps <= 300000:
            lr=self.initial_lr * (0.2 ** 3)
        cur_lr=lr
        for p in self.optimizer.param_groups:
            p['lr']=cur_lr
        return cur_lr

lr_scheduler=LrSchedule(initial_lr=learning_rate,optimizer=optimizer)

found = False
test_scenario_index=[2]
total_index=[a for a in range(len(train_loader_list))]
used_index=[item for item in total_index if item not in test_scenario_index]
print(used_index)
used_train_loader_list = [(train_loader_list[i], i) for i in used_index]
print(used_train_loader_list)
for epoch in range(epochs):
    random.shuffle(used_train_loader_list)
    for kk, (train_loader, dataset_idx) in enumerate(used_train_loader_list):
        for i, (data, local) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            data = data * 10000
            data = DFT_tensor(num_ant=ant_size, num_car=car_size, data=data.float(), F_delay=F_delay,
                               F_angle=F_angle)
            data = data.cuda()
            local = local.cuda().float()
            local_current = local[:, 0:2]
            local_pred = model(data, dataset_idx)
            MSE_loss = nn.MSELoss()(local_pred, local_current)
            loss = MSE_loss
            loss.backward()
            lr_scheduler.step()
            if lr_scheduler.steps <= steps:
                optimizer.step()
            else:
                found = True
                break
        if found == True:
            break
    if found == True:
        break

    if epoch % 20== 0:
        print('lr:%.4e' % optimizer.param_groups[0]['lr'])
        print("epoch :", epoch)
        print("step :", lr_scheduler.steps)
        torch.save(model.state_dict(), './model' + '.pth')

        model.eval()
        for kk, test_loader in enumerate(test_loader_list):
            sum_rmse = 0
            for i, (data, local) in enumerate(test_loader):
                optimizer.zero_grad()
                data = data * 10000
                data = DFT_tensor(num_ant=ant_size, num_car=car_size, data=data.float(), F_delay=F_delay,
                                  F_angle=F_angle)
                data = data.cuda()
                local = local.cuda().float()
                local_current = local[:, 0:2]
                with torch.no_grad():
                    local_pred = model(data,kk)
                rmse = RMSE(local_pred, local_current)
                sum_rmse+=rmse
            avg_rmse=sum_rmse/(i+1)
            print("scenario index:", kk)
            print("avg_distance :", avg_rmse)
torch.save(model.state_dict(), './model' + '.pth')