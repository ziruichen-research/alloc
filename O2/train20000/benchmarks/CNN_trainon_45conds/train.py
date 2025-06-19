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
learning_rate =1e-3
print_freq = 200
hidden_size=32

car_size=32
ant_size=32

path_total='/home/chenzirui/Desktop/home_mnt/dynamicscenario_50scenarios_32ant_32car_40MHz_BS1_25.5.31'
path_scenario_index=[f'/scenario{6 + i * 20}' for i in range(50)]
path_list=[path_total+scenario_index+'/train20000' for scenario_index in path_scenario_index]
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

model=LocNet(num_ant=ant_size,num_car=car_size,hidden_size=hidden_size)
from calflops import calculate_flops
input1=torch.randn(1,ant_size,car_size,2)
inputs={
    "data":input1
}
flops, macs, params = calculate_flops(model=model, kwargs=inputs, print_results=True)
print("FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))

if len(gpu_list.split(',')) > 1:
    model = torch.nn.DataParallel(model).cuda()
else:
    model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

input = torch.rand([1, ant_size, car_size, 2]).cuda()
flops, params = profile(model, inputs=(input,))
print("#flops : ", flops)
print("params : ", params)
print('#model parameters:', sum(param.numel() for param in model.parameters()))

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
test_scenario_index=[item*5+1 for item in [1,2,5,7,8]]
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
            data=data*100000
            local=local.cuda().float()
            local_current=local[:,0:2]

            local_pred=model(data)
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

    if epoch % 3== 0:
        print('lr:%.4e' % optimizer.param_groups[0]['lr'])
        print("epoch :", epoch)
        print("step :", lr_scheduler.steps)
        torch.save(model.state_dict(), './model' + '.pth')

        model.eval()
        evaluated_scenario_index = test_scenario_index + [6 * 5 + 1]
        for kk, test_loader in enumerate(test_loader_list):
            if kk in evaluated_scenario_index:
                sum_rmse = 0
                for i, (data, local) in enumerate(test_loader):
                    optimizer.zero_grad()
                    data = data.cuda().float()
                    data = data * 100000
                    local = local.cuda().float()
                    local_current = local[:, 0:2]
                    with torch.no_grad():
                        local_pred = model(data)
                    rmse = RMSE(local_pred, local_current)
                    sum_rmse+=rmse
                avg_rmse=sum_rmse/(i+1)
                print("scenario index:", (kk-1)//5*100+26)
                print("avg_distance :", avg_rmse)
torch.save(model.state_dict(), './model' + '.pth')