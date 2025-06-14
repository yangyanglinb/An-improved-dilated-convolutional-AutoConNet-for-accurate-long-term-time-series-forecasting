#!/usr/bin/env bash
cd "$(dirname "$0")"/..

# —— 快速/完整 模式切换 —— 
FAST=${FAST:-1}
if (( FAST )); then
    EPOCH=1;  PAT=1;   BATCH=64;  LR=5e-4
    DM=32;    DFF=64;   EL=2
else
    EPOCH=50; PAT=10;  BATCH=32;  LR=5e-4
    DM=64;    DFF=128;  EL=3
fi

MODEL_B=AutoConNet
MODEL_I=AutoConNet_Improved

# —— 维护 compare_all.csv —— 
CSV=results/compare_all.csv
mkdir -p results
if [[ ! -f $CSV ]]; then
    echo "time,dataset,MSE_base,MSE_impr,MSE_drop(%),MAE_base,MAE_impr,MAE_drop(%),RMSE_base,RMSE_impr,RMSE_drop(%),MAPE_base,MAPE_impr,MAPE_drop(%)" > "$CSV"
fi

grab(){ grep -oP "$1:\s*\K[0-9.]+" "$2" | tail -1; }
rmse(){ awk -v x="$1" 'BEGIN{ if(x==""||x=="nan"){printf "0.000000";} else {printf "%.6f", sqrt(x);} }'; }
pct(){ awk -v b="$1" -v i="$2" 'BEGIN{ if(b==""||b=="0"||i==""){printf "0.00";} else {printf "%.2f", (b-i)*100/b;} }'; }

# —— 数据集 参数 —— 
DS=ETTm2
ROOT="/content/drive/MyDrive/Colab Notebooks/Self-Supervised-Contrasitive-Data/ETT-small"
FILE=ETTm2.csv
SL=336;  PL=96;  FT=S;  CI=1;  FREQ=h

TS=$(date +%Y%m%d_%H%M%S)
echo -e "\n==================  [ ${DS} ]  ==================\n"

# —— 公共 参数 数组 —— 
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
    --d_model       "$DM"
    --d_ff          "$DFF"
    --e_layers      "$EL"
    --batch_size    "$BATCH"
    --learning_rate "$LR"
    --patience      "$PAT"
    --train_ratio   0.6
    --freq          "$FREQ"
    --checkpoints   "./checkpoints"
)

# —— Baseline 训练 —— 
B_ID=${DS}_B_${TS}; B_LOG=log_${DS}_B.txt
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
I_ID=${DS}_I_${TS}; I_LOG=log_${DS}_I.txt
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

# —— 测试 & 保存 预测 —— 
echo -e "\n📤  Saving ${DS} predictions …\n"
mkdir -p results/$DS/{baseline,improved}

# Baseline 预测
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

# Improved 预测
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
B_MSE=$(grab mse  "$B_LOG");   B_MAE=$(grab mae  "$B_LOG")
B_MAPE=$(grab mape "$B_LOG");   B_RMSE=$(rmse "$B_MSE")
I_MSE=$(grab mse  "$I_LOG");   I_MAE=$(grab mae  "$I_LOG")
I_MAPE=$(grab mape "$I_LOG");   I_RMSE=$(rmse "$I_MSE")
MSE_D=$(pct "$B_MSE" "$I_MSE");  MAE_D=$(pct "$B_MAE" "$I_MAE")
RMSE_D=$(pct "$B_RMSE" "$I_RMSE"); MAPE_D=$(pct "$B_MAPE" "$I_MAPE")
echo "$TS,$DS,$B_MSE,$I_MSE,$MSE_D,$B_MAE,$I_MAE,$MAE_D,$B_RMSE,$I_RMSE,$RMSE_D,$B_MAPE,$I_MAPE,$MAPE_D" >> "$CSV"
echo -e "✔ [${DS}] done — appended to compare_all.csv\n"
