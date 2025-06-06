#!/usr/bin/env bash
# ------------------------------------------------------------
# scripts/train_all.sh  ——  batch run for all datasets
# ------------------------------------------------------------
cd "$(dirname "$0")"/..

set -e                                  # 整体出错即退出
mkdir -p results

# ========== 1. 数据集清单 ==========
NAMES=(ETTh1 ETTh2 ETTm1 ETTm2 EXR ILI WEA \
       M4H M4D M4M M4Q M4W M4Y \
       M5V M5E)

ROOTS=(./dataset/ETT-small ./dataset/ETT-small ./dataset/ETT-small ./dataset/ETT-small \
       ./dataset/exchange_rate ./dataset/illness ./dataset/weather \
       ./dataset/M4/Test ./dataset/M4/Test ./dataset/M4/Test ./dataset/M4/Test ./dataset/M4/Test ./dataset/M4/Test \
       ./dataset/M5 ./dataset/M5)

FILES=(ETTh1.csv ETTh2.csv ETTm1.csv ETTm2.csv \
       exchange_rate.csv national_illness.csv weather.csv \
       Hourly-test.csv Daily-test.csv Monthly-test.csv Quarterly-test.csv Weekly-test.csv Yearly-test.csv \
       sales_train_validation.csv sales_train_evaluation.csv)

# —— 序列长度 & 预测步长（可按需再调） ——
SEQL=(336 336 720 720 168  104 720 720 720 168 168 260  36 168 168)
PREDL=( 96  96 192 192  24  36 336 192 192  24  24  52   6  28  28)

# —— 特征类型 & enc/dec/c_out ——
FEATS=(S S S S S S M S S S S S S S S)
ENCIN=(1 1 1 1 1 1 21 1 1 1 1 1 1 1 1)           # weather=21 其余 1

# ========== 2. 通用 & 快速模式参数 ==========
FAST=${FAST:-1}              # 如果外部没传 FAST，就默认为 1（快速模式）；可在命令行里写 FAST=0 来切换到完整版
if (( FAST )); then
    EPOCH=3;  PAT=1;  BATCH=64;  LR=5e-4
    DM=32;    DFF=64; EL=2        # 快速模式：小模型、少轮数
else
    EPOCH=18; PAT=10; BATCH=32; LR=5e-4
    DM=64;    DFF=128; EL=3       # 正式模式：大模型、多轮数
fi

MODEL_B=AutoConNet
MODEL_I=AutoConNet_Improved
CSV=results/compare_all.csv
[[ ! -f $CSV ]] && echo "time,dataset,MSE_base,MSE_impr,MSE_drop(%),MAE_base,MAE_impr,MAE_drop(%),RMSE_base,RMSE_impr,RMSE_drop(%),MAPE_base,MAPE_impr,MAPE_drop(%)" > "$CSV"

# ========== 3. 工具函数 ==========
timestamp(){ date +%Y%m%d_%H%M%S; }
grab(){ grep -oP "$1:\s*\K[0-9.]+" "$2" | tail -1; }

# 纯 awk，不依赖 python —— 最高兼容性
rmse(){ awk -v x="${1:-nan}" 'BEGIN{printf "%.6f", (x=="nan"?0:sqrt(x))}'; }
pct(){  awk -v b="$1" -v i="$2" 'BEGIN{printf "%.2f", (b-i)*100/b}'; }

# ========== 4. 主循环 ==========
for i in "${!NAMES[@]}"; do
  DS=${NAMES[$i]}
  ROOT=${ROOTS[$i]}
  FILE=${FILES[$i]}
  SL=${SEQL[$i]}
  PL=${PREDL[$i]}
  FT=${FEATS[$i]}
  CI=${ENCIN[$i]}
  TS=$(timestamp)

  echo -e "\n==================  [ ${DS} ]  ==================\n"

  # ---- 公共超参串 ----
  COMMON="--root_path $ROOT --data_path $FILE --data $DS \
          --features $FT --enc_in $CI --dec_in $CI --c_out $CI \
          --seq_len $SL --label_len 48 --pred_len $PL \
          --d_model $DM --d_ff $DFF --e_layers $EL \
          --batch_size $BATCH --learning_rate $LR --patience $PAT \
          --train_ratio 0.6"            # 保证 len 非负

  TRAIN="--is_training 1 --train_epochs $EPOCH"
  TEST="--is_training 0 --save"

  # ---- 个性化覆写 ----
  case $DS in
    ILI        ) COMMON+=" --freq d" ;;           # Illness: Daily
    WEA        ) COMMON+=" --freq h" ;;           # Weather: Hourly (features=M)
    M5*        ) COMMON+=" --freq d" ;;           # M5: Daily

    M4H        ) COMMON+=" --freq h" ;;           # M4 Hourly
    M4D        ) COMMON+=" --freq d" ;;           # M4 Daily
    M4M        ) COMMON+=" --freq m" ;;           # M4 Monthly
    M4Q        ) COMMON+=" --freq q" ;;           # M4 Quarterly
    M4W        ) COMMON+=" --freq w" ;;           # M4 Weekly
    M4Y        ) COMMON+=" --freq y" ;;           # M4 Yearly
  esac

  # ========== 训练 Baseline ==========
  B_ID=${DS}_B_${TS}; B_LOG=log_${DS}_B.txt
  set +e                      # ↓↓↓ 允许单集失败
  python -u run.py $COMMON $TRAIN --model $MODEL_B --model_id $B_ID | tee "$B_LOG"
  [[ ${PIPESTATUS[0]} -ne 0 ]] && { echo "⚠ $DS Baseline failed, skip"; set -e; continue; }

  # ========== 训练 Improved ==========
  I_ID=${DS}_I_${TS}; I_LOG=log_${DS}_I.txt
  python -u run.py $COMMON $TRAIN --model $MODEL_I --model_id $I_ID | tee "$I_LOG"
  [[ ${PIPESTATUS[0]} -ne 0 ]] && { echo "⚠ $DS Improved failed, skip"; set -e; continue; }

  # ========== 预测并保存 ==========
  echo -e "\n📤  Saving ${DS} predictions …\n"
  rm -rf results/$DS 2>/dev/null; mkdir -p results/$DS/{baseline,improved}

  python -u run.py $COMMON $TEST --model $MODEL_B --model_id $B_ID
  mv results/long_term_forecast_${DS}_B_*/pred.npy  results/$DS/baseline/ 2>/dev/null
  mv results/long_term_forecast_${DS}_B_*/true.npy  results/$DS/baseline/ 2>/dev/null
  rm -rf results/long_term_forecast_${DS}_B_* 2>/dev/null

  python -u run.py $COMMON $TEST --model $MODEL_I --model_id $I_ID
  mv results/long_term_forecast_${DS}_I_*/pred.npy  results/$DS/improved/ 2>/dev/null
  mv results/long_term_forecast_${DS}_I_*/true.npy  results/$DS/improved/ 2>/dev/null
  rm -rf results/long_term_forecast_${DS}_I_* 2>/dev/null

  # ========== 提取指标并写 CSV ==========
  B_MSE=$(grab mse  "$B_LOG"); B_MAE=$(grab mae  "$B_LOG"); B_MAPE=$(grab mape "$B_LOG"); B_RMSE=$(rmse "$B_MSE")
  I_MSE=$(grab mse  "$I_LOG"); I_MAE=$(grab mae  "$I_LOG"); I_MAPE=$(grab mape "$I_LOG"); I_RMSE=$(rmse "$I_MSE")

  MSE_D=$(pct "$B_MSE"  "$I_MSE"); MAE_D=$(pct "$B_MAE"  "$I_MAE")
  RMSE_D=$(pct "$B_RMSE" "$I_RMSE"); MAPE_D=$(pct "$B_MAPE" "$I_MAPE")

  echo "$TS,$DS,$B_MSE,$I_MSE,$MSE_D,$B_MAE,$I_MAE,$MAE_D,$B_RMSE,$I_RMSE,$RMSE_D,$B_MAPE,$I_MAPE,$MAPE_D" >> "$CSV"
  echo -e "✔ [$DS] done — appended to compare_all.csv\n"
  set -e
done

echo -e "\n🏁  All datasets finished.\n"
