import torch
import torch.utils.data as Data
import torch.nn as nn
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

def kl_divergence(mu_q, logvar_q, mu_p, logvar_p):
    """
    计算两个独立高斯分布的逐元素 KL(q||p)，
    然后对最后一个隐变量维度 m 求和，最后对 batch 和节点维度求平均。
    输入 shape: (B, N, m)
    """
    kld_element = 0.5 * (
        logvar_p - logvar_q
        + (torch.exp(logvar_q) + (mu_q - mu_p).pow(2)) / torch.exp(logvar_p)
        - 1
    )  # (B, N, m)
    # sum over latent dim m → (B, N), then mean over B and N
    return kld_element.sum(dim=-1).mean()

def loss_fn(mu_q, logvar_q, mu_p, logvar_p, Y_base, R_hat, Y, Y_pred, beta=1.0):
    """
    参数：
      mu_q, logvar_q: Tensor (B,N,m)
      mu_p, logvar_p: Tensor (B,N,m)
      Y_base: Tensor (B,T,N,C)  —— 基准预测
      R_hat : Tensor (B,T,N,C)  —— 残差预测
      Y     : Tensor (B,T,N,C)  —— 真实值
      beta  : float              —— KL 项权重
    返回：
      loss: 标量 Tensor
    """
    # print("Y_base shape:", Y_base.shape)
    # print("Y shape:", Y.shape)  
    # 1) 基准分支 MSE：先不 reduce
    # mse_base = F.mse_loss(Y_base, Y, reduction='none')  # (B,T,N,C)
    # # mean over time, node, channel → (B,)
    # loss_base = mse_base.mean(dim=(1,2,3)).mean()       # scalar

    # # 2) 残差分支 MSE
    # residual_true = Y - Y_base.detach()
    # mse_res = F.mse_loss(R_hat, residual_true, reduction='none')
    # loss_res = mse_res.mean(dim=(1,2,3)).mean()

    loss_base = masked_mae_loss(Y,Y_pred)

    # 3) KL 散度
    loss_kl = kl_divergence(mu_q, logvar_q, mu_p, logvar_p) 

    # 总 loss
    return loss_base + beta * loss_kl


class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def normalization(data,ratio):
    # print(np.array(self.data.squeeze().mean()).shape)
    l = len(data)
    num = int(l * ratio)
    mean=np.array(data[:num]).squeeze().mean()
    std=np.array(data[:num]).squeeze().std() #[..., 0]
    scalar = StandardScaler(mean,std)
    data = scalar.transform(data)
    print("标准化成功，均值是：", mean, "方差是：", std)
    return data, scalar



def list2loader(data, batch_size, shuffle=True):
    input = torch.tensor(np.array(data[0]),dtype=torch.float32)
    # input2 = torch.tensor(np.array(data[1]),dtype=torch.float32)
    output = torch.tensor(np.array(data[1]),dtype=torch.float32).squeeze(-1)

    torch_data = Data.TensorDataset(input, output)
    loader = Data.DataLoader(dataset=torch_data,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                drop_last=True,
                                num_workers=2)
    return loader




def masked_mae_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    mask /= mask.mean()

    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    # loss[loss != loss] = 0
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    loss = loss.mean()
    return loss


 
def evaluation(labels, preds, null_val=np.nan):

    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
        mask = mask.float()
        mask /= torch.mean(mask)
        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    mape = torch.mean(loss)
    mape = mape.cpu()
    # mape = mape.mean().cpu()

    mae = torch.abs(labels - preds)
    mae = mae * mask
    # mae = torch.where(torch.isnan(mae), torch.zeros_like(mae), mae)
    mae = mae.mean().cpu()

    mse = torch.pow(labels - preds, 2)
    mse = mse * mask
    # rmse = torch.where(torch.isnan(rmse), torch.zeros_like(rmse), rmse)
    mse = mse.mean().cpu()

    return mae, mse, mape #mse



def pcc(x, y):
    x, y = x.reshape(-1).cpu(), y.reshape(-1).cpu()
    return np.corrcoef(x, y)[0][1]


