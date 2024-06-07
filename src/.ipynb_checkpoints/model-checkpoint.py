import numpy as np
from datetime import datetime
from arch import *
from metric import *
from datamodule import *
class Model:
    def __init__(
        self,
        net,
        path: str = "",
        batch_size: int = 8,
        learning_rate: float = 1e-3,
        init_lr: float = 1e-1,
        last_lr: float = 1e-5,
        warmup: int = 5,
        epochs: int = 50,
        device: str = 'cpu',
    ):
        self.net = net
        self.path = path
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.init_lr = init_lr
        self.last_lr = last_lr
        self.warmup = warmup
        self.epochs = epochs
        self.device = device
        self.set_normalize()
        self.set_denormalize()
    def set_normalize(self):
        self.mean_radar = np.load(os.path.join(self.path,'normalize_mean_radar.npy'))
        self.std_radar = np.load(os.path.join(self.path,'normalize_std_radar.npy'))
        self.mean_satellite = np.load(os.path.join(self.path,'x_normalize_mean_satellite.npy'))
        self.std_satellite = np.load(os.path.join(self.path,'x_normalize_std_satellite.npy'))
    def set_denormalize(self):
        self.denormalize_radar = transforms.Normalize(-self.mean_radar/self.std_radar,1/self.std_radar)
        self.denormalize_satellite = transforms.Normalize(-self.mean_satellite/self.std_satellite,1/self.std_satellite)
    def data_prepare(self):
        path_train = os.path.join(self.path,'train')
        train_list_dir = self.load_file(path = path_train)
        print('Number of train sample: %d' % (len(train_list_dir)))
        self.train_loader = self.data_loader(list_dir=train_list_dir,isShuffle=True)
        path_val = os.path.join(self.path,'val')
        val_list_dir = self.load_file(path = path_val)
        print('Number of val sample: %d' % (len(val_list_dir)))
        self.val_loader = self.data_loader(list_dir=val_list_dir,isShuffle=False)
        path_test = os.path.join(self.path,'test')
        test_list_dir = self.load_file(path = path_test)
        print('Number of test sample: %d' % (len(test_list_dir)))
        self.test_loader = self.data_loader(list_dir=test_list_dir,isShuffle=False)
    def load_file(self,path):
        pred_radar_dir =os.path.join(path,"predict_radar")
        pred_satellite_dir = os.path.join(path,"x_predict_24_2")
        GT_radar_dir = os.path.join(path ,"GT_radar")
        GT_satellite_dir = os.path.join(path,"x_GT_24_2")
        list_dir = []
        for name in os.listdir(pred_radar_dir):
            temp = []
            temp.append(os.path.join(pred_radar_dir,name))
            pred_satellite_path = os.path.join(pred_satellite_dir,name[0:-6]+name[-4:])
            GT_radar_path = os.path.join(GT_radar_dir, name)
            GT_satellite_path = os.path.join(GT_satellite_dir, name[0:-6] + name[-4:])
            if(os.path.isfile(pred_satellite_path)):
                temp.append(pred_satellite_path)
            if(os.path.isfile(GT_radar_path)):
                temp.append(GT_radar_path)
            if(os.path.isfile(GT_satellite_path)):
                temp.append(GT_satellite_path)
            if(len(temp) == 4):
                list_dir.append(temp)
        return list_dir
    def data_loader(self,list_dir, isShuffle:bool):
        transform_radar = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean_radar,self.std_radar)
        ])
        transform_satellite = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(20),
            transforms.Normalize(self.mean_satellite,self.std_satellite)
        ])
        dataset = CustomDataset(list_img_dir=list_dir, transform_radar=transform_radar,transform_satellite = transform_satellite)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=isShuffle)
        return data_loader
    def save_weight(self):
        checkpoint = "checkpoint/"
        os.makedirs(checkpoint, exist_ok=True)
        i = 0
        while(os.path.exists(checkpoint +f"model_weights_{i}.pth")):
            i+=1
        path = checkpoint + f"model_weights_{i}.pth"
        torch.save(self.net.state_dict(),path)
    def val(self):
        val_loss_radar_avg, val_loss_satellite_avg, num_batches = 0,0, 0
        for i, data in enumerate(self.val_loader, 0):
            with torch.no_grad():
                inp_radar,inp_satellite,out_radar,out_satellite = data
                inp_radar,inp_satellite = inp_radar.to(self.device),inp_satellite.to(self.device)
                out_radar,out_satellite = out_radar.to(self.device),out_satellite.to(self.device)
                pred_radar,pred_satellite = self.net(inp_radar,inp_satellite)
                loss_radar = F.mse_loss(pred_radar,out_radar)
                loss_satellite = F.mse_loss(pred_satellite,out_satellite)
                val_loss_radar_avg += loss_radar.item()
                val_loss_satellite_avg += loss_satellite.item()
                num_batches += 1
        val_loss_radar_avg /= num_batches
        val_loss_satellite_avg /= num_batches
        print('Val loss radar: %.7f' % ( val_loss_radar_avg))
        print('Val loss satellite: %.7f' % ( val_loss_satellite_avg))
    def train(self):
        criterion = nn.MSELoss()
        optimizer = optim.SGD(self.net.parameters(), lr=self.init_lr, momentum=0.9)
        T_max = self.epochs
        T_cur = 0
        lr_list = [0]
        self.net.to(self.device)
        print('Start Training!!!!')
        for epoch in range(1, self.epochs+1):  # loop over the dataset multiple times
            running_loss = 0.0
            epoch_steps = 0
            T_cur += 1
            # warm-up
            if epoch <= self.warmup:
                optimizer.param_groups[0]['lr'] = (1.0 * epoch) / self.warmup  * self.init_lr
            else:
                # cosine annealing lr
                optimizer.param_groups[0]['lr'] = self.last_lr + (self.init_lr - self.last_lr) * (1 + np.cos(T_cur * np.pi / T_max)) / 2

            for i, data in enumerate(self.train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inp_radar,inp_satellite,out_radar,out_satellite = data
                inp_radar,inp_satellite = inp_radar.to(self.device),inp_satellite.to(self.device)
                out_radar,out_satellite = out_radar.to(self.device),out_satellite.to(self.device)
                # forward + backward + optimize
                pred_radar,pred_satellite = self.net(inp_radar,inp_satellite)
                optimizer.zero_grad()
                loss = criterion(pred_radar, out_radar)
                #loss.backward()
                loss = criterion(pred_satellite, out_satellite)
                loss.backward()
                optimizer.step()
                # print statistics
                running_loss += loss.item()
                epoch_steps += 1
                if i + 1 == len(self.train_loader):
                    print("[Epoch % d]: "% (epoch), datetime.now(),sep ="")
                    print("Total loss: %.7f" % (running_loss / epoch_steps))
                    print("Learning Rate: %.7f" % (lr_list[-1]))
                    running_loss = 0.0
            lr_list.append(optimizer.param_groups[0]['lr'])
            if(self.val_loader != None): self.val()
        print("Finished Training")
    def test(self):
        self.net.to(self.device)
        self.net.eval()
        metrics = {MSE,RMSE}
        loss = {}
        num_batches = 0
        for met in metrics: 
            loss[met] = {'radar':0.0,'satellite':0.0}
        for i, data in enumerate(self.test_loader, 0):
            with torch.no_grad():
                inp_radar,inp_satellite,out_radar,out_satellite = data
                inp_radar,inp_satellite = inp_radar.to(self.device),inp_satellite.to(self.device)
                out_radar,out_satellite = out_radar.to(self.device),out_satellite.to(self.device)
                pred_radar,pred_satellite = self.net(inp_radar,inp_satellite)
                out_radar = self.denormalize_radar(out_radar)
                pred_radar = self.denormalize_radar(pred_radar)
                out_satellite = self.denormalize_satellite(out_satellite)
                pred_satellie = self.denormalize_satellite(pred_satellite)
                for met in metrics:
                    loss[met]['radar'] += met(pred_radar,out_radar)
                    loss[met]['satellite'] += met(pred_satellite,out_satellite)
                num_batches += 1
        for met in metrics:
            test_loss_radar_avg = loss[met]['radar']/num_batches
            test_loss_satellite_avg = loss[met]['satellite']/num_batches
            print(f'{met.__name__} loss radar: %f' % ( test_loss_radar_avg))
            print(f'{met.__name__} loss satellite: %f' % ( test_loss_satellite_avg))

