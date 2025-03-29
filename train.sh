#!/bin/bash

# 인자 설정
DATA="gsm_math"
LOAD=true
EMBED_MODEL="sentence-transformers/all-MiniLM-L6-v2"

BATCH_SIZE=1024
CVAE_EPOCHS=1000
CVAE_LR=1e-4
RATIONALE_EPOCHS=1000
RATIONALE_LR=1e-4
OUTPUT="./output"

Z_DIM=128
KL_WEIGHT=7.0
RATIONALE_METHOD="kld"
DEVICE="cuda"

LOG_FILE="train_${DATA}_${RATIONALE_METHOD}_${KL_WEIGHT}_z${Z_DIM}.log"

# 변수 설정 로그 출력
{
  echo "=== Training CVAE Model ==="
  echo "DATA: $DATA"
  echo "LOAD: $LOAD"
  echo "EMBED_MODEL: $EMBED_MODEL"
  echo "Z_DIM: $Z_DIM"
  echo "BATCH_SIZE: $BATCH_SIZE"
  echo "CVAE_EPOCHS: $CVAE_EPOCHS"
  echo "CVAE_LR: $CVAE_LR"
  echo "KL_WEIGHT: $KL_WEIGHT"
  echo "RATIONALE_METHOD: $RATIONALE_METHOD"
  echo "RATIONALE_EPOCHS: $RATIONALE_EPOCHS"
  echo "RATIONALE_LR: $RATIONALE_LR"
  echo "OUTPUT: $OUTPUT"
  echo "DEVICE: $DEVICE"
  echo "==========================="
} > "./log/${LOG_FILE}"

# 실행
nohup python train.py \
    --data "$DATA" \
    --load "$LOAD" \
    --embed_model "$EMBED_MODEL" \
    --z_dim $Z_DIM \
    --batch_size $BATCH_SIZE \
    --cvae_epochs $CVAE_EPOCHS \
    --cvae_lr $CVAE_LR \
    --kl_weight $KL_WEIGHT \
    --rationale_train_method "$RATIONALE_METHOD" \
    --rationale_epochs $RATIONALE_EPOCHS \
    --rationale_lr $RATIONALE_LR \
    --output "$OUTPUT" \
    --device "$DEVICE" \
    > "./log/${LOG_FILE}" 2>&1 &
