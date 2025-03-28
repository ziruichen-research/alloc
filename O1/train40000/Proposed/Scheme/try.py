import os
gpu_list = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
import numpy as np
import math
import torch
import torch.nn as nn
import random

input = torch.zeros(3, 5)
target = torch.zeros(3, 5)+1
loss=nn.MSELoss(reduction='')(input,target)
print(loss)