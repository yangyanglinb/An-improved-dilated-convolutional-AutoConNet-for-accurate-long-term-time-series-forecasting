# data_provider/data_factory.py
# ------------------------------------------------------------
# 统一的数据加载入口：根据 args.data 取 Dataset ，返回 (Dataset, DataLoader)
# ------------------------------------------------------------
import os, pandas as pd
from torch.utils.data import DataLoader
from data_provider.data_loader import (
    Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom
)

# ------------------------------------------------------------
# 1. 数据键到 Dataset 类的映射
# ------------------------------------------------------------
data_dict = {
    # ---------- ETT ----------
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,

    # ---------- 经典公共 ----------
    'custom':           Dataset_Custom,
    'electricity':      Dataset_Custom,
    'traffic':          Dataset_Custom,
    'weather':          Dataset_Custom,
    'exchange_rate':    Dataset_Custom,
    'national_illness': Dataset_Custom,

    # ---------- 自定义缩写 ----------
    'EXR': Dataset_Custom,   # exchange_rate.csv
    'ILI': Dataset_Custom,   # national_illness.csv
    'WEA': Dataset_Custom,   # weather.csv

    # ---------- M4 ----------
    'M4H': Dataset_Custom, 'M4D': Dataset_Custom, 'M4M': Dataset_Custom,
    'M4Q': Dataset_Custom, 'M4W': Dataset_Custom, 'M4Y': Dataset_Custom,

    # ---------- M5 ----------
    'M5V': Dataset_Custom,   # validation
    'M5E': Dataset_Custom,   # evaluation
}

# ------------------------------------------------------------
# 2. 若 target 列不存在，自动兜底
# ------------------------------------------------------------
def _safe_target(root, file, wanted: str):
    """返回最终应使用的列名"""
    csv_cols = pd.read_csv(os.path.join(root, file), nrows=0).columns.tolist()
    if wanted in csv_cols:
        return wanted
    for cand in ('value', 'target'):
        if cand in csv_cols:
            return cand
    # 最后兜底：用最后一列
    return csv_cols[-1]

# ------------------------------------------------------------
# 3. 主入口
# ------------------------------------------------------------
def data_provider(args, flag: str):
    if args.data not in data_dict:
        keys = ', '.join(sorted(data_dict))
        raise KeyError(f"[data_provider] 未找到数据键 '{args.data}'. "
                       f"可选键：{keys}")

    DataCls = data_dict[args.data]
    timeenc = 1 if args.embed == 'timeF' else 0

    # ---------- DataLoader 参数 ----------
    if flag == 'test':
        shuffle_flag, batch_size = False, (
            1 if args.model == 'TimesNet' and args.task_name != 'anomaly_detection'
            else args.batch_size)
    else:  # train / val
        shuffle_flag, batch_size = True, args.batch_size

    # M4 / M5 不丢尾
    drop_last = not (args.data.startswith('M4') or args.data.startswith('M5'))

    # ---------- 处理 target ----------
    real_target = _safe_target(args.root_path, args.data_path, args.target)

    # ---------- 构造 Dataset ----------
    dataset = DataCls(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=real_target,
        timeenc=timeenc,
        freq=args.freq,
        seasonal_patterns=args.seasonal_patterns,
        train_ratio=args.train_ratio
    )
    print(f"{flag:5s}  len={len(dataset):>6}")

    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=shuffle_flag,
                        num_workers=args.num_workers,
                        drop_last=drop_last)
    return dataset, loader
