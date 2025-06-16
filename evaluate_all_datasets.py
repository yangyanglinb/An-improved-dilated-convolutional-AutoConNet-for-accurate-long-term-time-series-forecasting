import numpy as np
import os
import json
import pandas as pd
import matplotlib.pyplot as plt

def smape(y_true, y_pred, clip_value=120, smooth_window=5):
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    smape_values = 100 * numerator / (denominator + 1e-8)
    smape_values = np.clip(smape_values, 0, clip_value)
    smape_values = pd.Series(smape_values).rolling(window=smooth_window, min_periods=1).mean().values
    return np.mean(smape_values)

def mase(y_true, y_pred, y_train, seasonality=1):
    mae_model = np.mean(np.abs(y_pred - y_true))
    mae_naive = np.mean(np.abs(y_train[seasonality:] - y_train[:-seasonality]))
    return mae_model / (mae_naive + 1e-8)

def load_data(dataset, method):
    pred_path = f"results/{dataset}/{method}/pred.npy"
    true_path = f"results/{dataset}/{method}/true.npy"
    if not os.path.exists(pred_path) or not os.path.exists(true_path):
        print(f"⚠️ Missing files for {dataset}/{method}, skipped.")
        return None, None
    y_pred = np.load(pred_path)
    y_true = np.load(true_path)
    return y_true.flatten(), y_pred.flatten()

def clean_old_results():
    files_to_remove = [
        "results/metrics_summary.csv",
        "results/metrics_summary.json",
        "results/smape_comparison_improved.png",
        "results/mase_comparison_improved.png"
    ]
    for file in files_to_remove:
        if os.path.exists(file):
            os.remove(file)
            print(f"🗑️ Removed old file: {file}")
    print("✅ Old output files cleaned.\n")

def plot_charts(df):
    x = np.arange(len(df["Dataset"].unique()))
    width = 0.25  # 稍微增宽柱体
    gap = 0.06    # 增大柱组间距，避免两组柱子太近
    datasets = df["Dataset"].unique()

    def add_labels(bars, fmt):
        for bar in bars:
            height = bar.get_height()
            offset = height * 0.07  # 增大偏移量，避免数值挤在柱顶
            if np.isfinite(height):
                plt.text(bar.get_x() + bar.get_width() / 2, height + offset,
                         fmt.format(height),
                         ha='center', va='bottom', fontsize=7, rotation=0)

    def adjust_ylim(values, scale_factor=1.5):
        max_val = values.max()
        return max_val * scale_factor

    # sMAPE 图
    plt.figure(figsize=(22, 6), dpi=300)
    base_smape = df[df["Method"] == "baseline"]["sMAPE (%)"]
    imp_smape = df[df["Method"] == "improved"]["sMAPE (%)"]
    bars1 = plt.bar(x - (width / 2 + gap / 2), base_smape, width, label='Baseline', color='gray')
    bars2 = plt.bar(x + (width / 2 + gap / 2), imp_smape, width, label='Improved', color='royalblue')
    plt.yscale('log')
    plt.ylim(1, adjust_ylim(pd.concat([base_smape, imp_smape])))
    plt.xticks(x, datasets, rotation=45, ha='right')
    plt.ylabel("sMAPE (%) (Lower is better)")
    plt.title("sMAPE Comparison [log scale]")
    plt.legend()
    add_labels(bars1, "{:.1f}")
    add_labels(bars2, "{:.1f}")
    plt.tight_layout()
    plt.savefig("results/smape_comparison_improved.png")
    plt.close()

    # MASE 图
    plt.figure(figsize=(22, 6), dpi=300)
    base_mase = df[df["Method"] == "baseline"]["MASE"]
    imp_mase = df[df["Method"] == "improved"]["MASE"]
    bars1 = plt.bar(x - (width / 2 + gap / 2), base_mase, width, label='Baseline', color='gray')
    bars2 = plt.bar(x + (width / 2 + gap / 2), imp_mase, width, label='Improved', color='royalblue')
    plt.yscale('log')
    plt.ylim(1e-2, adjust_ylim(pd.concat([base_mase, imp_mase])))
    plt.xticks(x, datasets, rotation=45, ha='right')
    plt.ylabel("MASE (Lower is better)")
    plt.title("MASE Comparison [log scale]")
    plt.legend()
    add_labels(bars1, "{:.3f}")
    add_labels(bars2, "{:.3f}")
    plt.tight_layout()
    plt.savefig("results/mase_comparison_improved.png")
    plt.close()

    print("✅ Charts saved with enhanced spacing and layout.")

def main():
    clean_old_results()
    datasets = [
        "ETTh1", "ETTh2", "ETTm1", "ETTm2",
        "M4D", "M4H", "M4M", "M4Q", "M4W", "M4Y",
        "M5V", "M5E", "illness", "exchange_rate", "traffic", "weather"
    ]
    methods = ["baseline", "improved"]
    seasonality = 1
    results = []
    for dataset in datasets:
        for method in methods:
            print(f"\nEvaluating {dataset} - {method}")
            y_true, y_pred = load_data(dataset, method)
            if y_true is None or y_pred is None:
                continue
            if len(y_true) != len(y_pred):
                min_len = min(len(y_true), len(y_pred))
                y_true = y_true[-min_len:]
                y_pred = y_pred[-min_len:]
                print(f"⚠️ Cropped to {min_len} for alignment.")
            y_train = y_true
            m = mase(y_true, y_pred, y_train, seasonality=seasonality)
            s_raw = smape(y_true, y_pred)
            s_weighted = s_raw * (1 + m)
            results.append({
                "Dataset": dataset,
                "Method": method,
                "sMAPE (%)": float(round(s_weighted, 3)),
                "MASE": float(round(m, 5))
            })
            print(f"  sMAPE (weighted): {s_weighted:.3f}%, MASE: {m:.5f}")
    df = pd.DataFrame(results)
    df.to_csv("results/metrics_summary.csv", index=False)
    with open("results/metrics_summary.json", "w") as f_json:
        json.dump(results, f_json, indent=4)
    plot_charts(df)

if __name__ == "__main__":
    main()
