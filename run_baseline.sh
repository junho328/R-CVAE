#!/bin/bash

# 인자 설정
MODEL="meta-llama/Llama-3.1-8B-Instruct" # "Qwen/Qwen2.5-14B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"
DATA="EleutherAI/hendrycks_math" # "openai/gsm8k", "EleutherAI/hendrycks_math"
METHOD="top_p"
BATCH_SIZE=32
MAX_NEW_TOKENS=512
BEAM_WIDTH=8
TOP_P=0.95
TEMPERATURE=0.8

# 로그 파일명 설정
LOG_MODEL="llama"
LOG_DATA="math"
LOG_FILE="baseline_${LOG_MODEL}_${LOG_DATA}_${METHOD}.log"

# 변수 설정 로그 출력
{
  echo "=== Inference Baselines ==="
  echo "MODEL: $MODEL"
  echo "DATA: $DATA"
  echo "METHOD: $METHOD"
  echo "BATCH_SIZE: $BATCH_SIZE"
  echo "MAX_NEW_TOKENS: $MAX_NEW_TOKENS"
  echo "BEAM_WIDTH: $BEAM_WIDTH"
  echo "TOP_P: $TOP_P"
  echo "TEMPERATURE: $TEMPERATURE"
  echo "==========================="
} > "./log/${LOG_FILE}"

# 실행
nohup python baseline_inference.py \
    --model "$MODEL" \
    --data "$DATA" \
    --method "$METHOD" \
    --batch_size $BATCH_SIZE \
    --max_new_tokens $MAX_NEW_TOKENS \
    --beam_width $BEAM_WIDTH \
    --top_p $TOP_P \
    --temperature $TEMPERATURE \
    > "./log/${LOG_FILE}" 2>&1 &

