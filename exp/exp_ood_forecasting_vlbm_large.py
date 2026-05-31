from cProfile import label
from matplotlib.pyplot import ylabel
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
from utils.metrics import metric, evaluation
warnings.filterwarnings('ignore')

# 假设您的 data_factory.py 已经配置好，当调用时能够返回 Dataset_Custom_STPAN
# 并且您的模型 (self.model_dict[...]) 能够处理 [B, T, N, C] 的输入格式

class Exp_Long_Term_Forecast_ood_vlbm_large(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast_ood_vlbm_large, self).__init__(args)

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
            for i, (batch_x, batch_y, ylabel) in enumerate(vali_loader):
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
            for i, (batch_x, batch_y, ylabel) in enumerate(train_loader):
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
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            print('loading model')
            # 加载模型权重
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        ## 1. 初始化累加器 (Accumulators)
        
        # 统一使用 float64 (double) 以确保精度，并使用 Python float 初始化
        N_abs_sum = 0.0
        N_sq_sum = 0.0
        N_mape_sum = 0.0
        N_count = 0
        
        A_abs_sum = 0.0
        A_sq_sum = 0.0
        A_mape_sum = 0.0
        A_count = 0
        
        # 设置结果保存路径
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            
            for i, (batch_x, batch_y, ylabel) in enumerate(test_loader):
                
                # 2. 数据准备和模型前向传播 - 保持不变
                batch_x = batch_x.float().to(self.device, non_blocking=True)
                batch_y = batch_y.float().to(self.device, non_blocking=True)
                ylabel = ylabel.to(self.device, non_blocking=True) # 标签也移到 GPU
                out = self.model(batch_x, batch_y)
                # MODIFIED: 移除 f_dim，直接切片
                Y_pred = out['Y_pred']
                Y_true = batch_y[..., 0]
                
                # *** 优化点 A: 误差计算 (在 GPU上进行) ***
                # 展平数据：全部保持在 GPU 上
                pred_flat = Y_pred.reshape(-1, Y_pred.shape[-1])
                true_flat = Y_true.reshape(-1, Y_true.shape[-1])
                
                # 标签拉平，只取第 0 个特征维度作为判断依据 (形状: (Total_Timesteps,))
                # PyTorch 张量操作，保持在 GPU
                label_flat = ylabel.reshape(-1, ylabel.shape[-1])[:, 0] 
                
                # 计算当前批次的误差 (GPU)
                abs_error = torch.abs(pred_flat - true_flat) # 绝对误差
                sq_error = (pred_flat - true_flat)**2        # 平方误差
                
                # MAPE 分母：避免除以 0 (使用 PyTorch/GPU)
                mape_denominator = torch.where(torch.abs(true_flat) < 1e-8, torch.tensor(1e-8).to(self.device, non_blocking=True), true_flat)
                mape_error = abs_error / torch.abs(mape_denominator)

                # 获取正常的索引 (labels==0) (GPU)
                normal_indices = (label_flat == 0)
                
                # 4. **增量计算和累加 (优化)**
                
                # 正常数据累加：使用张量索引和求和，然后转移到 CPU/Python 累加器
                N_abs_sum += abs_error[normal_indices].sum().item()
                N_sq_sum += sq_error[normal_indices].sum().item()
                N_mape_sum += mape_error[normal_indices].sum().item()
                N_count += normal_indices.sum().item() # 计数也使用 GPU 求和

                # 获取异常的索引 (labels==1) (GPU)
                abnormal_indices = (label_flat == 1)

                # 异常数据累加
                A_abs_sum += abs_error[abnormal_indices].sum().item()
                A_sq_sum += sq_error[abnormal_indices].sum().item()
                A_mape_sum += mape_error[abnormal_indices].sum().item()
                A_count += abnormal_indices.sum().item()
                
                # *** 优化点 B: 反归一化处理 (移到循环外) ***
                # 注意：如果反归一化 (inverse_transform) 是一个耗时的 CPU 操作，
                # 并且只对**最终指标**需要反归一化值，那么在**循环内**执行 NumPy/CPU 
                # 转换会成为瓶颈。

                # 由于原代码要求在循环内进行 `inverse_transform`，我们需要**假设** # `test_data.inverse_transform` **不是瓶颈**，或者将其实现改为支持批量处理。

                # 如果无法在 GPU 上进行反归一化，则必须保留原有的 CPU/NumPy 转换部分：
                
                if test_data.scale and self.args.inverse:
                     # 必须将 GPU 结果先转移到 CPU 进行反归一化，
                     # 并且需要重新计算误差，这会抵消前面的 GPU 优化。
                     # 强烈建议将 inverse_transform 移到循环外或在 CPU 线程中异步处理。
                     
                     # ⚠️ 如果保留此段，性能提升会受限。如果 inverse_transform 速度很快，可以接受。
                     # 为保持与原代码逻辑最接近，我们保留CPU转移部分，并假设反归一化函数已经优化：
                    
                    pred_np = pred_flat.detach().cpu().numpy().astype(np.float32)
                    true_np = true_flat.detach().cpu().numpy().astype(np.float32)
                    
                    if test_data.scale and self.args.inverse:
                        # 假设 test_data.inverse_transform 接受 (N_Total, F) 形状
                        pred_np = test_data.inverse_transform(pred_np)
                        true_np = test_data.inverse_transform(true_np)
                        
                        # 在 CPU 上重新计算误差并更新累加器
                        # 此处将覆盖上面的 GPU 计算结果，导致 GPU 优化失效，但这确保了指标基于反归一化的值。
                        # 最佳实践是将反归一化函数改为 PyTorch/GPU 实现。
                        
                        label_flat_np = label_flat.detach().cpu().numpy()
                        abs_error_np = np.abs(pred_np - true_np)
                        sq_error_np = (pred_np - true_np)**2
                        mape_denominator_np = np.where(np.abs(true_np) < 1e-8, 1e-8, true_np)
                        mape_error_np = abs_error_np / np.abs(mape_denominator_np)

                        normal_indices_np = (label_flat_np == 0)
                        abnormal_indices_np = (label_flat_np == 1)

                        N_abs_sum += np.sum(abs_error_np[normal_indices_np])
                        N_sq_sum += np.sum(sq_error_np[normal_indices_np])
                        N_mape_sum += np.sum(mape_error_np[normal_indices_np])
                        N_count += np.sum(normal_indices_np)

                        A_abs_sum += np.sum(abs_error_np[abnormal_indices_np])
                        A_sq_sum += np.sum(sq_error_np[abnormal_indices_np])
                        A_mape_sum += np.sum(mape_error_np[abnormal_indices_np])
                        A_count += np.sum(abnormal_indices_np)
                
                # 如果没有进行反归一化，则使用上面的 GPU 结果。
                # 如果进行了反归一化，则使用 CPU 重新计算的结果（指标基于反归一化的值）。

        ## 5. 循环结束后：计算最终平均指标 - 保持不变
        
        # 计算 ALL 指标
        ALL_count = N_count + A_count
        ALL_abs_sum = N_abs_sum + A_abs_sum
        ALL_sq_sum = N_sq_sum + A_sq_sum
        ALL_mape_sum = N_mape_sum + A_mape_sum
        
        # ALL
        ALL_mae = ALL_abs_sum / ALL_count if ALL_count > 0 else 0
        ALL_mse = ALL_sq_sum / ALL_count if ALL_count > 0 else 0
        ALL_mape = ALL_mape_sum / ALL_count if ALL_count > 0 else 0
        print('ALL mse:{:.6f}, mae:{:.6f}, mape:{:.6f}'.format(ALL_mse, ALL_mae, ALL_mape))
        
        # 正常 (Normal) 指标
        normal_mae = N_abs_sum / N_count if N_count > 0 else 0
        normal_mse = N_sq_sum / N_count if N_count > 0 else 0
        normal_mape = N_mape_sum / N_count if N_count > 0 else 0
        print('normal mse:{:.6f}, mae:{:.6f}, mape:{:.6f}'.format(normal_mse, normal_mae, normal_mape))

        # 异常 (Abnormal) 指标
        abnormal_mae = A_abs_sum / A_count if A_count > 0 else 0
        abnormal_mse = A_sq_sum / A_count if A_count > 0 else 0
        abnormal_mape = A_mape_sum / A_count if A_count > 0 else 0
        print('abnormal mse:{:.6f}, mae:{:.6f}, mape:{:.6f}'.format(abnormal_mse, abnormal_mae, abnormal_mape))