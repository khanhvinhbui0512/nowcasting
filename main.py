import os
import torch
import math
from src.metric import *
from torchvision import transforms 
import numpy as np
from copy import deepcopy
#mean = np.load('/data/data_WF/ERA5/0.25deg_24_npz/normalize_mean.npz')
#np.save('/data/data_WF/finetune/normalize_mean_satellite.npy',mean['total_precipitation'])
#std = np.load('/data/data_WF/ERA5/0.25deg_24_npz/normalize_std.npz')
#np.save('/data/data_WF/finetune/normalize_std_satellite.npy',std['total_precipitation'])
#path = '/data/data_WF/finetune/train/GT_radar/'
#all = []
#num = 0
#cnt = 0.0
#sum = 0.0
#var = 0.0
#mean = np.load('/data/data_WF/finetune/normalize_mean_radar.npy')
#print(mean)
#for name in os.listdir(path):
    #file = np.load(os.path.join(path,name))
    #old_cnt = deepcopy(cnt)
    #new_cnt = cnt + file.shape[0]*file.shape[1]
    #mean = ((new_cnt-old_cnt)/new_cnt)*np.mean(file) + (old_cnt/new_cnt)*mean
    #file = file - mean
    #ssq = np.sum(file**2)
    #var = ssq/new_cnt + (old_cnt/new_cnt)*var 
    #std = np.std(all)
    #np.save('/data/data_WF/finetune/normalize_mean_radar.npy',mean)
    #std = math.sqrt(var)
    #np.save('/data/data_WF/finetune/normalize_std_radar.npy',std)
    #cnt = new_cnt
    #num += 1
    #print("Number %d and Std %.5f" %( num,std),end='\r')
path = '/data/data_WF/finetune/'
mse = 0
cnt = 0
mean = 0.5
std = 0.5
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.CenterCrop(20)
])
mean_satellite = np.load(os.path.join(path,'x_normalize_mean_satellite.npy'))[0]
std_satellite = np.load(os.path.join(path,'x_normalize_std_satellite.npy'))[0]
norm =transforms.Compose([
    transforms.ToTensor(),
    transforms.CenterCrop(20),
    transforms.Normalize(mean_satellite,std_satellite)
])
path += 'test/'
denorm = transforms.Normalize(-mean_satellite/std_satellite,1/std_satellite)
for name in os.listdir(path + 'x_predict_24_2'):
    a = np.load(path+'x_predict_24_2/'+name)
    b = np.load(path+'x_GT_24_2/'+name)
    c = transform(a)
    d = transform(b)
    u = norm(a)
    v = norm(b)
    u = denorm(u)
    v = denorm(v)
    print(MSE(c,d))
    print(MSE(u,v))
    print(u)
    print(c)
    cnt += 1
    break
#print(mse/cnt)
