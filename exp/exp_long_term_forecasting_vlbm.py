from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from exp.vlbm_loss import loss_fn, masked_mae_loss
warnings.filterwarnings('ignore')

# 假设您的 data_factory.py 已经配置好，当调用时能够返回 Dataset_Custom_STPAN
# 并且您的模型 (self.model_dict[...]) 能够处理 [B, T, N, C] 的输入格式

class Exp_Long_Term_Forecast_vlbm(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast_vlbm, self).__init__(args)

    def _build_model(self):
        # 这里的模型需要能够处理 [B, T, N, C] 格式的输入
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        # 此处会调用您的 data_provider，返回 Dataset_Custom_STPAN 实例
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        return model_optim

    def _select_criterion(self, use="MSE"):
        if use == "MAE":
            criterion = nn.L1Loss()
        else: # 默认使用 MSE
            criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            # MODIFIED: 使用 _ 忽略 dataloader 返回的无用 mark 变量
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device, non_blocking=True)
                batch_y = batch_y.float().to(self.device, non_blocking=True) # batch_y 也先移到 device
                out= self.model(batch_x, batch_y)
                Y_pred = out['Y_pred']
                Y_true = batch_y[..., 0]
                loss = masked_mae_loss(Y_pred, Y_true)
                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion(self.args.loss)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            
            # MODIFIED: 使用 _ 忽略 mark 变量
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device, non_blocking=True)
                batch_y = batch_y.float().to(self.device, non_blocking=True)
                out = self.model(batch_x, batch_y)
                #暂时先不用inverse回数据空间
                Y_true = batch_y[..., 0]
                # Y_true = scalar.inverse_transform(batch_y[..., 0])
                # Extract
                # Y_pred = scalar.inverse_transform(out['Y_pred'])       # (B, T, N)
                Y_base = out['Y_base']                                # (B, T, N)
                R_hat  = out['R_hat']                                 # (B, T, N)
                mu_q, logvar_q = out['mu_q'], out['logvar_q']         # (B, N, m)
                mu_p, logvar_p = out['mu_p'], out['logvar_p']         # (B, N, m)
                loss = loss_fn(mu_q, logvar_q,
                               mu_p, logvar_p,
                               Y_base, R_hat,
                               Y_true, out['Y_pred'], beta=0.1)
                train_loss.append(loss.item())
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion) # test_loss 也用 vali 函数计算
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def test(self, setting, test=0):
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            # MODIFIED: 使用 _ 忽略 mark 变量
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device, non_blocking=True)
                batch_y = batch_y.float().to(self.device, non_blocking=True)
                out = self.model(batch_x, batch_y)
                # MODIFIED: 移除 f_dim，直接切片
                Y_pred = out['Y_pred']
                Y_true = batch_y[..., 0]
                outputs_cpu = Y_pred.detach().cpu()
                batch_y_cpu = Y_true.detach().cpu()
                # MODIFIED: 适配新的 inverse_transform，该方法现在处理4D张量并返回3D张量
                if test_data.scale and self.args.inverse:
                    # inverse_transform 在内部处理 reshaping, 返回 [B, T, N]
                    outputs_inversed = test_data.inverse_transform(outputs_cpu) 
                    batch_y_inversed = test_data.inverse_transform(batch_y_cpu)
                    preds.append(outputs_inversed)
                    trues.append(batch_y_inversed)
                else:
                    # 如果不反转，只取数值通道 [..., 0]
                    preds.append(outputs_cpu.numpy())
                    trues.append(batch_y_cpu.numpy())
                # if i % 200 == 0:
                #     input = batch_x[...,0:1].detach().cpu().numpy()
                #     # print(input.shape, batch_y_cpu.shape,outputs_cpu.shape)
                #     gt = np.concatenate((input[0, :, -1, 0], batch_y_cpu[0, :, -1]), axis=0)
                #     pd = np.concatenate((input[0, :, -1, 0], outputs_cpu[0, :, -1]), axis=0)
                #     visual(gt, pd, os.path.join(folder_path, str(i) + '.png'))

        # MODIFIED: 使用 np.concatenate 获得最终的预测和真实值数组
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}'.format(mse, mae, rmse, mape, mspe))
        # ... (保存结果的逻辑可以保持不变) ...
        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        return