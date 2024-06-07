import numpy as np
from torch.utils.data import DataLoader, Dataset
class CustomDataset(Dataset):
    def __init__(self, list_img_dir: list[list[str]], transform_radar = None ,transform_satellite=None):
        self.transform_radar = transform_radar
        self.transform_satellite = transform_satellite
        self.list_img_dir = list_img_dir
    def __len__(self):
        return len(self.list_img_dir)
    def __getitem__(self, idx):
        #print(self.list_img_dir[idx])
        inp_radar = np.load(self.list_img_dir[idx][0])
        inp_satellite = np.load(self.list_img_dir[idx][1])
        out_radar = np.load(self.list_img_dir[idx][2])
        out_satellite = np.load(self.list_img_dir[idx][3])
        if(self.transform_radar):
            inp_radar = self.transform_radar(inp_radar)
            out_radar = self.transform_radar(out_radar)
        if(self.transform_satellite):
            inp_satellite = self.transform_satellite(inp_satellite)
            out_satellite = self.transform_satellite(out_satellite)
        return inp_radar,inp_satellite.float(),out_radar, out_satellite.float()
