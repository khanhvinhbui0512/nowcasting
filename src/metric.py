import numpy as np
import torch 
def MSE(pred,GT,lat,clim):
    return torch.mean((pred-GT)**2)
def RMSE(pred,GT,lat,clim):
    return torch.sqrt(torch.mean((pred-GT)**2)) 
def MAE(pred,GT,lat,clim):
    return torch.mean(torch.abs(pred-GT))
def WMSE(pred, y, lat,clim):
    if(lat is None):return 0
    error = (pred - y) ** 2  # [N, C, H, W]
    # lattitude weights
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean() 
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=error.dtype, device=error.device)  # (1, H, 1)
    loss =  (error * w_lat).mean()
    return loss
def WRMSE(pred,GT,lat,clim):
    if(lat is None):return 0
    error = (pred - GT) ** 2  # [B, V, H, W]
    # lattitude weights
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=error.dtype, device=error.device)
    loss = torch.mean(
                torch.sqrt(torch.mean(error* w_lat, dim=(-2, -1)))
            )
    return loss
def ACC(pred,GT,lat,clim):
    if(lat is None):return 0
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=pred.dtype, device=pred.device)  # [1, H, 1]

    # clim = torch.mean(y, dim=(0, 1), keepdim=True)
    clim = clim.to(device=GT.device).unsqueeze(0)
    pred = pred - clim
    GT = GT - clim
    pred_prime = pred - torch.mean(pred)
    GT_prime = GT - torch.mean(GT)
    loss = torch.sum(w_lat * pred_prime * GT_prime) / torch.sqrt(
        torch.sum(w_lat * pred_prime**2) * torch.sum(w_lat * GT_prime**2)
    )
    return loss
