from model import *
import argparse
import torch
import numpy as np
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--usegpu',type=int,default=4)
    parser.add_argument('--batchsize',type=int,default=4)
    parser.add_argument('--path',type=str,default="")
    args = parser.parse_args()
    print("GPU is Available: ", torch.cuda.is_available())
    print("Number of GPUS: ",torch.cuda.device_count())
    device = 'cpu'
    if(torch.cuda.is_available()): device = f'cuda:{args.usegpu}'
    print(device)
    net = Unet()
    #net.load_state_dict(torch.load('checkpoint/model_weights_0.pth'))
    #net.to(device)
    net.eval()
    model = Model(net,
                  path=args.path,
                  batch_size=args.batchsize,
                  learning_rate = 1e-3,
                  init_lr = 1e-1,
                  last_lr = 1e-5,
                  warmup=5  ,              
                  epochs=50,
                  device=device,)
    model.data_prepare()
    model.train()
    model.save_weight()
    model.test()
if __name__ == "__main__":
    main()

