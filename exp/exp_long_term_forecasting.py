# exp/exp_long_term_forecasting.py
# ------------------------------------------------------------
# Self-Supervised-Contrastive-Forecasting – 长周期预测实验脚本
# ------------------------------------------------------------
import os, time, json, warnings
import numpy as np
import torch
import torch.nn as nn
from torch import optim

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric, shape_metric

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    """长周期多数据集统一训练/测试流程"""

    # --------------------------------------------------------
    def __init__(self, args):
        super().__init__(args)

    # --------------------------------------------------------
    # 1. 构建模型
    # --------------------------------------------------------
    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        print(f'model parameters:{self.count_parameters(model)}')
        return model

    # --------------------------------------------------------
    # 2. 数据加载
    # --------------------------------------------------------
    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    # --------------------------------------------------------
    # 3. 优化器 & 损失
    # --------------------------------------------------------
    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()

    # --------------------------------------------------------
    # 4. 验证
    # --------------------------------------------------------
    @torch.no_grad()
    def vali(self, _, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for _, batch_x, batch_y, batch_x_mark, batch_y_mark in vali_loader:
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :])
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp],
                                dim=1).float().to(self.device)

            # forward
            if self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

            total_loss.append(criterion(outputs, batch_y).item())

        self.model.train()
        return float(np.mean(total_loss))

    # --------------------------------------------------------
    # 5. 训练
    # --------------------------------------------------------
    def train(self, setting):
        logs = {'train_loss': [], 'val_loss': [], 'test_loss': []}

        train_data, train_loader = self._get_data('train')
        vali_data, vali_loader = self._get_data('val')
        test_data, test_loader = self._get_data('test')

        ckpt_dir = os.path.join(self.args.checkpoints, setting)
        os.makedirs(ckpt_dir, exist_ok=True)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None

        train_steps = len(train_loader)

        for epoch in range(self.args.train_epochs):
            epoch_start = time.time()
            self.model.train()
            losses = []

            for i, (_, batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :])
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp],
                                    dim=1).float().to(self.device)

                # forward
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                target = batch_y[:, -self.args.pred_len:, f_dim:]

                loss = criterion(outputs, target)
                losses.append(loss.item())

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                # 每 100 步打印一次
                if (i + 1) % 100 == 0:
                    spent = time.time() - epoch_start
                    speed = spent / (i + 1)
                    remain = speed * ((self.args.train_epochs - epoch - 1) * train_steps + (train_steps - i - 1))
                    print(f"\titers: {i+1}, epoch: {epoch+1} | loss: {loss.item():.7f}")
                    print(f"\tspeed: {speed:.4f}s/iter; left time: {remain:.1f}s")

            # 每个 epoch 结束后
            train_loss = float(np.mean(losses))
            val_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            logs['train_loss'].append(train_loss)
            logs['val_loss'].append(val_loss)
            logs['test_loss'].append(test_loss)

            print(f"Epoch: {epoch+1:>3} | Train {train_loss:.7f}  Val {val_loss:.7f}  Test {test_loss:.7f}"
                  f"  (cost {time.time() - epoch_start:.1f}s)")

            early_stopping(val_loss, self.model, ckpt_dir)
            if early_stopping.early_stop:
                print("Early stopping triggered!")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        # 加载最优权重
        self.model.load_state_dict(torch.load(os.path.join(ckpt_dir, 'checkpoint.pth')))
        json.dump(logs, open(os.path.join(ckpt_dir, 'loss_logs.json'), 'w'))
        return self.model

    # --------------------------------------------------------
    # 6. 测试 / 推理
    # --------------------------------------------------------
    @torch.no_grad()
    def test(self, setting, test: int = 0):
        test_data, test_loader = self._get_data('test')

        if test:  # 手动指定时加载 best ckpt
            print('loading model …')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints', setting, 'checkpoint.pth')))

        self.model.eval()
        preds, trues = [], []

        for _, batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)

            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :])
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp],
                                dim=1).float().to(self.device)

            if self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

            preds.append(outputs.cpu().numpy())
            trues.append(batch_y.cpu().numpy())

        # 使用 concatenate 而非 np.array
        if len(preds) > 0:
            preds = np.concatenate(preds, axis=0)  # (N, pred_len, C)
            trues = np.concatenate(trues, axis=0)
        else:
            # 如果没有任何样本，返回空数组
            preds = np.zeros((0, self.args.pred_len, self.args.c_out))
            trues = np.zeros((0, self.args.pred_len, self.args.c_out))

        print('test shape:', preds.shape, trues.shape)

        # 保存到 ./results/<setting>/
        folder = os.path.join('./results', setting)
        os.makedirs(folder, exist_ok=True)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        # dilate_e, shape_e, temporal_e = shape_metric(preds, trues)
        dilate_e = shape_e = temporal_e = 0.0

        print(f'mse:{mse:.6f}, mae:{mae:.6f}, mape:{mape:.6f}, mspe:{mspe:.6f}, '
              f'dilate:{dilate_e:.7f}, shapedtw:{shape_e:.7f}, temporaldtw:{temporal_e:.7f}')

        if self.args.save:
            np.save(os.path.join(folder, 'metrics.npy'), np.array([mae, mse, rmse, mape, mspe]))
            np.save(os.path.join(folder, 'pred.npy'), preds)
            np.save(os.path.join(folder, 'true.npy'), trues)

        return mse, mae, mape, mspe, dilate_e, shape_e, temporal_e
