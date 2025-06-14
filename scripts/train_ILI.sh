#!/usr/bin/env bash
cd "$(dirname "$0")"/..

# 快速模式（1）或完整模式（0）
FAST=${FAST:-0}
if (( FAST )); then
    EPOCH=1; PAT=1; BATCH=64; LR=5e-4; DM=32; DFF=64; EL=2
else
    EPOCH=50; PAT=10; BATCH=32; LR=5e-4; DM=64; DFF=128; EL=3
fi

MODEL_B=AutoConNet
MODEL_I=AutoConNet_Improved

CSV=results/compare_all.csv
mkdir -p results
if [[ ! -f $CSV ]]; then
    echo "time,dataset,MSE_base,MSE_impr,MSE_drop(%),MAE_base,MAE_impr,MAE_drop(%),RMSE_base,RMSE_impr,RMSE_drop(%),MAPE_base,MAPE_impr,MAPE_drop(%)" > "$CSV"
fi

DS=illness
ROOT="/content/drive/MyDrive/Colab Notebooks/Self-Supervised-Contrasitive-Data/illness"
FILE=national_illness.csv
SL=84
PL=12
FT=S
CI=1
FREQ=d

TS=$(date +%Y%m%d_%H%M%S)
echo -e "\n=====  [${DS}]  =====\n"

TRAIN_ARGS=(
    --root_path     "$ROOT"
    --data_path     "$FILE"
    --data          "$DS"
    --features      "$FT"
    --enc_in        "$CI"
    --dec_in        "$CI"
    --c_out         "$CI"
    --seq_len       "$SL"
    --label_len     48
    --pred_len      "$PL"
    --train_ratio   0.6
    --freq          "$FREQ"
    --checkpoints   "./checkpoints"
    --d_model       "$DM"
    --d_ff          "$DFF"
    --e_layers      "$EL"
    --batch_size    "$BATCH"
    --learning_rate "$LR"
    --patience      "$PAT"
)

# —— Baseline 训练 —— 
B_ID=${DS}_B_${TS}
B_LOG=log_${DS}_B.txt
python -u run.py \
    "${TRAIN_ARGS[@]}" \
    --is_training    1 \
    --train_epochs   "$EPOCH" \
    --model          "$MODEL_B" \
    --model_id       "$B_ID" \
    --use_gpu        True \
    --gpu            0 \
    --num_workers    4 \
    | tee "$B_LOG"

# —— Improved 训练 —— 
I_ID=${DS}_I_${TS}
I_LOG=log_${DS}_I.txt
python -u run.py \
    "${TRAIN_ARGS[@]}" \
    --is_training    1 \
    --train_epochs   "$EPOCH" \
    --model          "$MODEL_I" \
    --model_id       "$I_ID" \
    --use_attention \
    --use_gpu        True \
    --gpu            0 \
    --num_workers    4 \
    | tee "$I_LOG"

# —— 测试 & 保存预测 —— 
echo -e "\n📤  Saving ${DS} predictions …\n"
mkdir -p results/$DS/{baseline,improved}

# Baseline 保存
python -u run.py \
    "${TRAIN_ARGS[@]}" \
    --is_training    0 \
    --save \
    --model          "$MODEL_B" \
    --model_id       "$B_ID" \
    --use_gpu        True \
    --gpu            0 \
    --num_workers    4

mv results/long_term_forecast_${DS}_B_*/pred.npy results/$DS/baseline/ 2>/dev/null
mv results/long_term_forecast_${DS}_B_*/true.npy results/$DS/baseline/ 2>/dev/null
rm -rf results/long_term_forecast_${DS}_B_* 2>/dev/null

# Improved 保存
python -u run.py \
    "${TRAIN_ARGS[@]}" \
    --is_training    0 \
    --save \
    --model          "$MODEL_I" \
    --model_id       "$I_ID" \
    --use_attention \
    --use_gpu        True \
    --gpu            0 \
    --num_workers    4

mv results/long_term_forecast_${DS}_I_*/pred.npy results/$DS/improved/ 2>/dev/null
mv results/long_term_forecast_${DS}_I_*/true.npy results/$DS/improved/ 2>/dev/null
rm -rf results/long_term_forecast_${DS}_I_* 2>/dev/null

# —— 写入 compare_all.csv —— 
grab(){ grep -oP "$1:\s*\K[0-9.]+" "$2" | tail -1; }
rmse(){ awk -v x="$1" 'BEGIN{ printf "%.6f", sqrt(x+0); }'; }
pct(){ awk -v b="$1" -v i="$2" 'BEGIN{ if(b==0||i==""||b==""){printf "0.00";} else {printf "%.2f", (b-i)*100/b;} }'; }

B_MSE=$(grab mse  "$B_LOG");   B_MAE=$(grab mae  "$B_LOG");   B_MAPE=$(grab mape "$B_LOG");   B_RMSE=$(rmse "$B_MSE")
I_MSE=$(grab mse  "$I_LOG");   I_MAE=$(grab mae  "$I_LOG");   I_MAPE=$(grab mape "$I_LOG");   I_RMSE=$(rmse "$I_MSE")

MSE_D=$(pct "$B_MSE" "$I_MSE"); MAE_D=$(pct "$B_MAE" "$I_MAE")
RMSE_D=$(pct "$B_RMSE" "$I_RMSE"); MAPE_D=$(pct "$B_MAPE" "$I_MAPE")

echo "$TS,$DS,$B_MSE,$I_MSE,$MSE_D,$B_MAE,$I_MAE,$MAE_D,$B_RMSE,$I_RMSE,$RMSE_D,$B_MAPE,$I_MAPE,$MAPE_D" >> "$CSV"
echo -e "✔ [${DS}] done — appended to compare_all.csv\n"
