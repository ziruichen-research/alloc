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

batch_size=500
epochs=100000000
steps=300000
learning_rate=1e-4
print_freq=200

ant_size=32
car_size=32
hidden_size=256

depth=8

model=Nlocnet(ant_size=ant_size,car_size=car_size,embed_dim=hidden_size,num_heads=4,hidden_dim=hidden_size*2,depth=depth)
from calflops import calculate_flops
input1=torch.randn(1,64,ant_size,car_size,2)
input2=torch.randn(1,64,2)
input3=torch.randn(1,1,ant_size,car_size,2)
inputs={
    "neigh_channel":input1,
    "neigh_pos":input2,
    "current_channel":input3
}
flops, macs, params = calculate_flops(model=model, kwargs=inputs, print_results=False)
print("FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))


if len(gpu_list.split(',')) > 1:
    model = torch.nn.DataParallel(model).cuda()
else:
    model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



path_list=['/home/chenzirui/Desktop/home_mnt/multiplescenario_bothside/32ant*32car_3.5GHz_40MHz_BS2_R101-700_V1_24.5.26/train20000','/home/chenzirui/Desktop/home_mnt/multiplescenario_bothside/32ant*32car_3.5GHz_40MHz_BS3_R701-1200_V1_24.5.26/train20000',
           '/home/chenzirui/Desktop/home_mnt/multiplescenario_bothside/32ant*32car_3.5GHz_40MHz_BS5_R1201-1600_V1_24.5.26/train20000','/home/chenzirui/Desktop/home_mnt/multiplescenario_bothside/32ant*32car_3.5GHz_40MHz_BS8_R1601-2000_V1_24.5.26/train20000',
           '/home/chenzirui/Desktop/home_mnt/multiplescenario_bothside/32ant*32car_3.5GHz_40MHz_BS9_R2001-2700_V1_24.5.26/train20000']
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

def find_sequence_test(num,data,pos,pos_array,path_now,error_range=1):

    error=np.zeros_like(pos)
    error[0:2]=np.random.uniform(-1 * error_range, error_range, size=2)
    pos=pos+error
    current_pos=repeat(pos,'l -> train_num l',train_num=pos_array.shape[0])
    distance_array=np.linalg.norm(current_pos-pos_array,axis=1)
    indices = (np.argpartition(distance_array, num)[:num]).tolist()
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


test_num=20000
total_neighbor_num=64
least_neighbor_num=16
whole_index=[a for a in range(least_neighbor_num+1,total_neighbor_num+1+1)]
found = False
test_scenario_index=[2]
total_index=[a for a in range(len(train_loader_list))]
used_index=[item for item in total_index if item not in test_scenario_index]
print(used_index)
used_train_loader_list=[train_loader_list[i] for i in used_index]
for epoch in range(epochs):
    random.shuffle(used_train_loader_list)
    for kk, train_loader in enumerate(used_train_loader_list):
        for i, (data,local) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            data=data.cuda().float()
            data=data*10000
            local=local.cuda().float()
            local=local[:,:,0:2]

            index = [b for b in range(total_neighbor_num + 1)]
            random.shuffle(index)
            index_len=random.choice(whole_index)
            index=index[:index_len]
            data_neigh = data[:, index[:-1], ...]
            data_current = data[:, index[-1:], ...]
            local_neigh = local[:, index[:-1], ...]
            local_current = local[:, index[-1:], ...]

            local_pred=model(data_neigh,local_neigh,data_current)
            MSE_loss = nn.MSELoss()(local_pred, local_current)
            loss = MSE_loss
            loss.backward()
            lr_scheduler.step()
            if lr_scheduler.steps <= steps:
                optimizer.step()
            else:
                found =True
                break
        if found == True:
            break
    if found ==True:
        break


    if epoch % 20 == 0:
        print('lr:%.4e' % optimizer.param_groups[0]['lr'])
        print("epoch :",epoch)
        print("step :", lr_scheduler.steps)
        torch.save(model.state_dict(), './model' + '.pth')

        model.eval()
        for kk, test_loader in enumerate(test_loader_list):
            sum_rmse = 0
            for i, (data, local) in enumerate(test_loader):
                data = data.cuda().float()
                data = data * 10000
                local = local.cuda().float()
                local = local[:, :, 0:2]
                data_neigh = data[:, 0:-1, ...]
                data_current = data[:, -1:, ...]
                local_neigh = local[:, 0:-1, ...]
                local_current = local[:, -1:, ...]
                with torch.no_grad():
                    local_pred = model(data_neigh, local_neigh, data_current)
                rmse = RMSE(local_pred[:,-1,:], local_current[:,-1,:])
                sum_rmse+=rmse
            avg_rmse=sum_rmse/(i+1)
            print("scenario index:",kk)
            print("avg_distance :",avg_rmse)

torch.save(model.state_dict(), './model' + '.pth')