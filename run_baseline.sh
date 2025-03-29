#!/bin/bash

# 인자 설정
DATA="EleutherAI/hendrycks_math" # "openai/gsm8k", "EleutherAI/hendrycks_math", "HuggingFaceH4/MATH-500" , "Maxwell-Jia/AIME_2024"
MODEL="Qwen/Qwen2.5-14B-Instruct" # "HuggingFaceTB/SmolLM2-1.7B-Instruct", "meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen2.5-14B-Instruct", "mistralai/Mistral-Small-24B-Instruct-2501"
METHOD="top_p" # greedy, top-p, best
BATCH_SIZE=64
MAX_NEW_TOKENS=1024 # 512, 1024
BEAM_WIDTH=8
TOP_P=0.95
TEMPERATURE=0.8

# 로그 파일명 설정
LOG_MODEL="mistral"
LOG_DATA="math"
LOG_FILE="baseline_${LOG_MODEL}_${METHOD}_${LOG_DATA}_${MAX_NEW_TOKENS}tokens.log"

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

