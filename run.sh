#!/bin/bash

# 인자 설정
DATA="openai/gsm8k" # "openai/gsm8k", "EleutherAI/hendrycks_math"
MODEL="meta-llama/Llama-3.1-8B-Instruct" # Qwen/Qwen2.5-14B-Instruct, "meta-llama/Llama-3.1-8B-Instruct"
CVAE_CKPT="./output/checkpoint/mse_1.0_model_train.pth"
EMBED_MODEL="sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE=32
NUM_SAMPLES=16
MAX_NEW_TOKENS=256
METHOD="mse"
OUTPUT="./output/generated_answer/"

# 로그 파일명 생성
LOG_MODEL="llama"
LOG_DATA="gsm"
LOG_FILE="cvae_${LOG_MODEL}_${LOG_DATA}_${METHOD}.log"

{
  echo "=== Inference Baselines ==="
  echo "MODEL: $MODEL"
  echo "DATA: $DATA"
  echo "CVAE_CKPT: $CVAE_CKPT"
  echo "EMBED_MODEL: $EMBED_MODEL"
  echo "MAX_NEW_TOKENS: $MAX_NEW_TOKENS"
  echo "BATCH_SIZE: $BATCH_SIZE"
  echo "NUM_SAMPLES: $NUM_SAMPLES"
  echo "METHOD: $METHOD"
  echo "OUTPUT: $OUTPUT"
  echo "==========================="
} > "./log/${LOG_FILE}"

# 실행
nohup python inference.py \
    --data "$DATA" \
    --model "$MODEL" \
    --cvae_ckpt "$CVAE_CKPT" \
    --embed_model "$EMBED_MODEL" \
    --batch_size $BATCH_SIZE \
    --num_samples $NUM_SAMPLES \
    --max_new_tokens $MAX_NEW_TOKENS \
    --method "$METHOD" \
    --output "$OUTPUT" \
    > "./log/${LOG_FILE}" 2>&1 &
