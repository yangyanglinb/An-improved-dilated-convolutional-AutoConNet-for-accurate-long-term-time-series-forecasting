import numpy as np
import torch
from utils.dtwloss.dilate_loss import dilate_loss

# 防止除零和空数组
EPS = 1e-8

def RSE(pred, true):
    if pred.size == 0:
        return np.nan
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2) + EPS)

def CORR(pred, true):
    if pred.size == 0:
        return np.nan
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0))**2 * (pred - pred.mean(0))**2).sum(0) + EPS)
    return (u / d).mean(-1)

def MAE(pred, true):
    if pred.size == 0:
        return np.nan
    return np.mean(np.abs(pred - true))

def MSE(pred, true):
    if pred.size == 0:
        return np.nan
    return np.mean((pred - true)**2)

def RMSE(pred, true):
    mse = MSE(pred, true)
    return np.sqrt(mse) if not np.isnan(mse) else np.nan

def MAPE(pred, true):
    if pred.size == 0:
        return np.nan
    return np.mean(np.abs(pred - true) / (np.abs(true) + EPS)) * 100

def MSPE(pred, true):
    if pred.size == 0:
        return np.nan
    return np.mean(((pred - true) / (np.abs(true) + EPS))**2) * 100

def metric(pred, true):
    mae  = MAE(pred, true)
    mse  = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    return mae, mse, rmse, mape, mspe

def shape_metric(pred, true, batch_size=100):
    if pred.shape[0] == 0:
        return np.nan, np.nan, np.nan

    pred_t = torch.tensor(pred).cuda()
    true_t = torch.tensor(true).cuda()
    n_total = pred.shape[0]

    dilate_e = shape_e = temp_e = 0.0
    for st in range(0, n_total, batch_size):
        ed = min(st + batch_size, n_total)
        with torch.no_grad():
            d_e, s_dtw, t_dtw = dilate_loss(pred_t[st:ed], true_t[st:ed], 0.5, 0.01, 'cuda')
            dilate_e += d_e.cpu().item() * (ed - st)
            shape_e  += s_dtw.cpu().item() * (ed - st)
            temp_e   += t_dtw.cpu().item() * (ed - st)

    return dilate_e / n_total, shape_e / n_total, temp_e / n_total
