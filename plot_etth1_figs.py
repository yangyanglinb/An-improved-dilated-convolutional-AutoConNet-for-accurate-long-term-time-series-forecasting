import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- 固定短路径 ----------
BASE_DIR = "results/ETTh1/baseline"
IMPR_DIR = "results/ETTh1/improved"
assert os.path.exists(f"{BASE_DIR}/pred.npy"), "Baseline pred.npy not found"
assert os.path.exists(f"{IMPR_DIR}/pred.npy"), "Improved pred.npy not found"

pred_b = np.load(f"{BASE_DIR}/pred.npy")
pred_i = np.load(f"{IMPR_DIR}/pred.npy")
true   = np.load(f"{IMPR_DIR}/true.npy")   # true 相同即可

# ---------- 创建输出目录 ----------
OUT_DIR = "results/etth1_fig"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- 1. 预测曲线 ----------
idx = 0
t = range(pred_b.shape[1])
plt.figure(figsize=(8,3))
plt.plot(t, true[idx,:,0], label='True')
plt.plot(t, pred_b[idx,:,0], '--', label='Baseline')
plt.plot(t, pred_i[idx,:,0], '-.', label='Improved')
plt.xlabel("Time step"); plt.ylabel("Value")
plt.title("ETTh1 Forecast (sample 0)")
plt.legend(); plt.tight_layout()
plt.savefig(f"{OUT_DIR}/prediction_curve.png", dpi=300)
plt.close()

# ---------- 2. 提升百分比柱状图 ----------
df = pd.read_csv('results/compare_etth1.csv')
vals = df.iloc[-1][['MSE_drop(%)','MAE_drop(%)','RMSE_drop(%)','MAPE_drop(%)']].astype(float).values
plt.figure(figsize=(5,3))
plt.bar(['MSE','MAE','RMSE','MAPE'], vals)
plt.ylabel('Improvement (%)')
plt.title("ETTh1 – Improvement over Baseline")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/improvement_bar.png", dpi=300)
plt.close()

print(f"✔ Figures saved to {OUT_DIR}/")
