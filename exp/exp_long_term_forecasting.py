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

    def __init__(self, args):
        # 兼容别名：将 'illness' 映射到 ILI 数据集
        if getattr(args, 'data', None) == 'illness':
            args.data = 'ILI'
        # 根据训练集动态设置输入/输出维度
        train_dataset, _ = data_provider(args, 'train')
        if hasattr(train_dataset, 'data_x') and train_dataset.data_x.ndim >= 2:
            n_feats = train_dataset.data_x.shape[-1]
            args.enc_in = n_feats
            args.dec_in = n_feats
            args.c_out = n_feats
        super().__init__(args)

    def _build_model(self):
        model_cls = self.model_dict.get(self.args.model)
        if not model_cls:
            raise ValueError(f"Model {self.args.model} is not registered!")
        model = model_cls.Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        print(f'model parameters:{self.count_parameters(model)}')
        return model

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()

    @torch.no_grad()
    def vali(self, _, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for _, batch_x, batch_y, batch_x_mark, batch_y_mark in vali_loader:
            batch_x      = batch_x.float().to(self.device)
            batch_y      = batch_y.float().to(self.device)
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)

            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :])
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).to(self.device)

            if self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            f_dim   = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            target  = batch_y[:,   -self.args.pred_len:, f_dim:]

            # 如果最后一个 batch 的真实长度 < pred_len，就用最后一条重复填充
            T = target.size(1)
            if T < self.args.pred_len:
                pad  = self.args.pred_len - T
                last = target[:, -1:, :].detach()
                target = torch.cat([target, last.repeat(1, pad, 1)], dim=1)

            loss = criterion(outputs, target)
            # 跳过 NaN
            if torch.isnan(loss):
                continue
            total_loss.append(loss.item())

        self.model.train()
        if not total_loss:
            # 全部被跳过时，用一个较大的惩罚性 loss
            return 1e3
        return float(np.mean(total_loss))

    def train(self, setting):
        logs = {'train_loss': [], 'val_loss': [], 'test_loss': []}

        train_data, train_loader = self._get_data('train')
        vali_data, vali_loader   = self._get_data('val')
        test_data, test_loader   = self._get_data('test')

        ckpt_dir = os.path.join(self.args.checkpoints, setting)
        os.makedirs(ckpt_dir, exist_ok=True)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        optimizer      = self._select_optimizer()
        criterion      = self._select_criterion()
        scaler         = torch.cuda.amp.GradScaler() if self.args.use_amp else None

        for epoch in range(self.args.train_epochs):
            start = time.time()
            self.model.train()
            losses = []

            for i, (_, batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                optimizer.zero_grad()
                batch_x      = batch_x.float().to(self.device)
                batch_y      = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :])
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).to(self.device)

                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim   = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                target  = batch_y[:,   -self.args.pred_len:, f_dim:]
                loss    = criterion(outputs, target)
                losses.append(loss.item())

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                if (i + 1) % 100 == 0:
                    elapsed = time.time() - start
                    print(f"	iters: {i+1}, epoch: {epoch+1} | loss: {loss.item():.7f}")

            train_loss = float(np.mean(losses))
            val_loss   = self.vali(vali_data, vali_loader, criterion)
            test_loss  = self.vali(test_data, test_loader, criterion)
            logs['train_loss'].append(train_loss)
            logs['val_loss'].append(val_loss)
            logs['test_loss'].append(test_loss)

            print(f"Epoch {epoch+1} | Train {train_loss:.6f}  Val {val_loss:.6f}  Test {test_loss:.6f}  "
                  f"(time {time.time()-start:.1f}s)")

            early_stopping(val_loss, self.model, ckpt_dir)
            if early_stopping.early_stop:
                break
            adjust_learning_rate(optimizer, epoch+1, self.args)

        # load best
        self.model.load_state_dict(torch.load(os.path.join(ckpt_dir, 'checkpoint.pth')))
        json.dump(logs, open(os.path.join(ckpt_dir, 'loss_logs.json'), 'w'))
        return self.model

    @torch.no_grad()
    def test(self, setting, do_test: int = 0):
        """
        对测试集做预测并计算指标。遇到空 batch（outputs 第一维为 0）时跳过，避免最后拼接成全 nan。
        """
        test_data, test_loader = self._get_data('test')

        # 如果是从训练切换到测试，需要先加载 checkpoint
        if do_test:
            ckpt_path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
            self.model.load_state_dict(torch.load(ckpt_path))

        self.model.eval()

        preds, trues = [], []
        for _, batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
            batch_x      = batch_x.float().to(self.device)
            batch_y      = batch_y.float().to(self.device)
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)

            # 构造解码器输入
            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :])
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).to(self.device)

            # 前向
            if self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            # 如果这一批没有任何样本（e.g. batch_size=0），跳过
            if outputs.shape[0] == 0:
                continue

            # 取最后 pred_len 且按 MS/ M 切分特征通道
            f_dim   = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            target  = batch_y[:,   -self.args.pred_len:, f_dim:]

            # 如果有些序列本身短于 pred_len，用最后一个值补齐
            T = target.size(1)
            tgt_np = target.cpu().numpy()
            if T < self.args.pred_len:
                pad  = self.args.pred_len - T
                last = tgt_np[:, -1:, :]
                tgt_np = np.concatenate([tgt_np, np.repeat(last, pad, axis=1)], axis=1)

            preds.append(outputs.cpu().numpy())
            trues.append(tgt_np)

        # 拼接所有非空 batch，若全为空则返回严格的空数组
        if preds:
            preds = np.concatenate(preds, axis=0)
            trues = np.concatenate(trues, axis=0)
        else:
            print("⚠️  Warning: no valid test samples found, returning empty predictions")
            preds = np.zeros((0, self.args.pred_len, self.args.c_out))
            trues = np.zeros((0, self.args.pred_len, self.args.c_out))

        print('test shape:', preds.shape, trues.shape)

        # 计算并打印指标
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print(f"mse:{mse:.6f}, mae:{mae:.6f}, mape:{mape:.6f}, mspe:{mspe:.6f}")

        # 保存结果
        if self.args.save:
            folder = os.path.join('./results', setting)
            os.makedirs(folder, exist_ok=True)
            np.save(os.path.join(folder, 'metrics.npy'), np.array([mae, mse, rmse, mape, mspe]))
            np.save(os.path.join(folder, 'pred.npy'), preds)
            np.save(os.path.join(folder, 'true.npy'), trues)

        return mse, mae, mape, mspe, 0.0, 0.0, 0.0
