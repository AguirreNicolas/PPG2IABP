import torch
from torch.utils.data import TensorDataset, Dataset
from torch.utils.data import DataLoader
import numpy as np

    
class Dataset_V5(Dataset):
  def __init__(self,X_signal,X_ci,Y,Y_mask,final_len_x,diff_window_ppg,device):
    self.X_signal = X_signal
    self.X_ci = X_ci
    self.Y = Y
    self.Y_mask = Y_mask
    self.final_len_x = final_len_x
    self.diff_window_ppg = diff_window_ppg
    self.device = device

  def __getitem__(self, index):
    x = self.X_signal[index]
    x_ci = self.X_ci[index]
    y = self.Y[index]
    y_mask = self.Y_mask[index]


    # Desplazamos la señal
    idx_roll = np.random.randint(0, self.diff_window_ppg)
    x = x[:,idx_roll:idx_roll+self.final_len_x]

    x_s = x[0:1,:]
    # Por ultimo normalizamos [0, 1] a la señal
    a_max = np.amax(x_s, axis=1)
    a_min = np.amin(x_s, axis=1)
    x_s = (x_s - a_min[None,:]) / (a_max[None,:] - a_min[None,:])
    x_d1 = x[1:2,:]
    a_max = np.amax(x_d1, axis=1)
    a_min = np.amin(x_d1, axis=1)
    x_d1 = (x_d1 - a_min[None,:]) / (a_max[None,:] - a_min[None,:])

    x = np.concatenate((x_s,x_d1))

    x = torch.from_numpy(x).float().to(self.device)
    x_ci = torch.from_numpy(x_ci).float().to(self.device)
    
    y = torch.from_numpy(y).float().to(self.device)
    y_mask = torch.from_numpy(y_mask).float().to(self.device)

    return (x,x_ci,y,y_mask)
    
  def __len__(self):
    return len(self.X_signal)
    
    
class Dataset_V6(Dataset):
  def __init__(self,X_signal,Y,Y_mask,final_len_x,diff_window_ppg,device):
    self.X_signal = X_signal
    self.Y = Y
    self.Y_mask = Y_mask
    self.final_len_x = final_len_x
    self.diff_window_ppg = diff_window_ppg
    self.device = device

  def __getitem__(self, index):
    x = self.X_signal[index]
    y = self.Y[index]
    y_mask = self.Y_mask[index]


    # Desplazamos la señal
    idx_roll = np.random.randint(0, self.diff_window_ppg)
    x = x[:,idx_roll:idx_roll+self.final_len_x]

    x_s = x[0:1,:]
    # Por ultimo normalizamos [0, 1] a la señal
    a_max = np.amax(x_s, axis=1)
    a_min = np.amin(x_s, axis=1)
    x_s = (x_s - a_min[None,:]) / (a_max[None,:] - a_min[None,:])
    x_d1 = x[1:2,:]
    a_max = np.amax(x_d1, axis=1)
    a_min = np.amin(x_d1, axis=1)
    x_d1 = (x_d1 - a_min[None,:]) / (a_max[None,:] - a_min[None,:])

    x = np.concatenate((x_s,x_d1))

    x = torch.from_numpy(x).float().to(self.device)
    y = torch.from_numpy(y).float().to(self.device)
    y_mask = torch.from_numpy(y_mask).float().to(self.device)

    return (x,y,y_mask)
    
  def __len__(self):
    return len(self.X_signal)
