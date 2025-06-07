#!/usr/bin/env bash
cd "$(dirname "$0")"/..

FAST=${FAST:-1}
if (( FAST )); then
    EPOCH=1; PAT=1; BATCH=64; LR=5e-4; DM=32; DFF=64; EL=2
else
    EPOCH=50; PAT=10; BATCH=32; LR=5e-4; DM=64; DFF=128; EL=3
fi

MODEL_B=AutoConNet
MODEL_I=AutoConNet_Improved

CSV=results/compare_all.csv; mkdir -p results
if [[ ! -f $CSV ]]; then
    echo "time,dataset,..." > "$CSV"
fi

grab(){ grep -oP "$1:\s*\K[0-9.]+" "$2" | tail -1; }
rmse(){ awk -v x="$1" 'BEGIN{ if(x==""||x=="nan"){printf "0.000000";} else {printf "%.6f", sqrt(x);} }'; }
pct(){ awk -v b="$1" -v i="$2" 'BEGIN{ if(b==""||b=="0"||i==""){printf "0.00";} else {printf "%.2f", (b-i)*100/b;} }'; }

DS=M4D
DATA_ROOT=${DATA_ROOT:-"/content/drive/MyDrive/Colab Notebooks/Self-Supervised-Contrasitive-Data"}
ROOT="$DATA_ROOT/M4D/Test"
FILE=M4_Test.csv
SL=336; PL=96; FT=S; CI=1; FREQ=h

TS=$(date +%Y%m%d_%H%M%S)
echo -e "\n=====  [${DS}]  =====\n"

TRAIN_ARGS=( … 同上 … --checkpoints "./checkpoints" )

# —— 完全同上——
