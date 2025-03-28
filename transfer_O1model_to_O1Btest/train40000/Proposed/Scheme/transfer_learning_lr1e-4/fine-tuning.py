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

import torch.backends.cudnn as cudnn
seed = 1
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
cudnn.deterministic = True
cudnn.benchmark = False
def seed_worker(worker_id):
    np.random.seed(seed)
    random.seed(seed)

batch_size=500

ant_size=32
car_size=32
hidden_size=256

learning_rate=1e-4

depth=8

model=Nlocnet(ant_size=ant_size,car_size=car_size,embed_dim=hidden_size,num_heads=4,hidden_dim=hidden_size,depth=depth)

path_model='/home/chenzirui/Desktop/ALLoc/O1/train40000/Proposed/Scheme'
path_test_origin='/home/chenzirui/Desktop/home_mnt/32ant*32car_3.5GHz_40MHz_R501-1400_V1_24.4.12/train40000'
path_test='/home/chenzirui/Desktop/home_mnt/32ant*32car_3.5GHz_40MHz_O1B_R501-1400_V1_24.4.12/train40000'


if len(gpu_list.split(',')) > 1:
    model = torch.nn.DataParallel(model).cuda()
else:
    model = model.cuda()

model.load_state_dict(torch.load(path_model+'/model' +'.pth'),strict=False)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

test_dataset_origin=DatasetFolder(path_test_origin+'/sequence_test_len64')
test_loader_origin=torch.utils.data.DataLoader(
    test_dataset_origin, batch_size=batch_size, shuffle=True, num_workers=4,pin_memory=True, drop_last=True,worker_init_fn=seed_worker)

test_dataset=DatasetFolder(path_test+'/sequence_test_len64')
test_loader=torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True, num_workers=4,pin_memory=True, drop_last=True,worker_init_fn=seed_worker)

train_dataset = DatasetFolder(path_test+'/sequence')
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,pin_memory=True, drop_last=True,worker_init_fn=seed_worker)

tuning_epoch=15
total_neighbor_num=64
least_neighbor_num=16
whole_index=[a for a in range(least_neighbor_num,total_neighbor_num+1)]
test_origin_list=[]
test_list=[]
for epoch in range(tuning_epoch):

    print("epoch", epoch)
    sum_rmse = 0
    model.eval()
    for i, (data, local) in tqdm(enumerate(test_loader_origin),total=len(test_loader_origin)):
        model.eval()
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
        rmse = rmse.item()
        sum_rmse+=rmse
    avg_rmse=sum_rmse/(i+1)
    print("origin avg_distance :",avg_rmse)
    test_origin_list.append(avg_rmse)

    sum_rmse = 0
    rmse_distribution_list=[]
    model.eval()
    for i, (data, local) in tqdm(enumerate(test_loader),total=len(test_loader)):
        model.eval()
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
        rmse = rmse.item()
        sum_rmse+=rmse
    avg_rmse=sum_rmse/(i+1)
    print("other avg_distance :",avg_rmse)
    test_list.append(avg_rmse)

    for i, (data, local) in enumerate(train_loader):
        model.train()
        optimizer.zero_grad()
        data = data.cuda().float()
        data = data * 10000
        local = local.cuda().float()
        local = local[:, :, 0:2]

        index_neigh = random.sample([b for b in range(total_neighbor_num + 1)], random.choice(whole_index))
        index_current = random.sample([b for b in range(total_neighbor_num + 1)],
                                      random.choice([a for a in range(1, total_neighbor_num + 1)]))
        data_neigh = data[:, index_neigh, ...]
        data_current = data[:, index_current, ...]
        local_neigh = local[:, index_neigh, ...]
        local_current = local[:, index_current, ...]

        local_pred = model(data_neigh, local_neigh, data_current)
        MSE_loss = nn.MSELoss()(local_pred, local_current)
        loss = MSE_loss
        loss.backward()
        optimizer.step()

test_origin_array=np.array(test_origin_list)
test_array=np.array(test_list)
np.save('error_origin_array.npy',test_origin_array)
np.save('error_array.npy',test_array)