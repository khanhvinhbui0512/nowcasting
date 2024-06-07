import os
import numpy as np

def out_x(std,path,save_dir):
    for name in os.listdir(path):
        temp = np.load(path+name)
        temp = temp*std
        np.save(save_dir + name,temp)

def out_exp(std,path,save_dir):
    for name in os.listdir(path):
        temp = np.load(path+name)
        temp = 1 + np.maximum(temp*std,0)
        temp = np.log(temp)
        np.save(save_dir + name,temp)
path= '/data/data_WF/finetune/test/exp_predict_24_2/'
save_dir = '/data/data_WF/finetune/output/exp_pred/'
std = np.load('/data/data_WF/ERA5/0.25deg_24_exp/normalize_std.npz')['total_precipitation']
out_exp(std,path,save_dir)
