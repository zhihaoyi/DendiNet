import torch
import torch.nn.functional as F
import torch.nn.MSELoss as mse

def rmse(predictions, targets):
    return torch.sqrt(F.mse_loss(predictions, targets))
def mse(predictions, targets):
    return F.mse_loss(predictions, targets)
def rmse_tsf(t1, t2, rmse=True):
    ##[forecast_window, batch], we compute rmse as computing rmse of each batch and then take avg over whole batchs
    batch_num=t1.shape[1]
   
    loss=torch.nn.MSELoss()
   
    if rmse is True:
        rmse = torch.empty((batch_num))
        for i in range(batch_num):
            rmse[i] = torch.sqrt(loss(t1[:,i],t2[:,i]))
        return torch.mean(rmse)
    else:
        mse = torch.empty((batch_num))
        for i in range(batch_num):
            mse[i] = loss(t1[:,i],t2[:,i])
            return torch.mean(mse)
