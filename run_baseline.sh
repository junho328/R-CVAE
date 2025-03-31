#!/bin/bash

# 인자 설정
DATA="openai/gsm8k" # "openai/gsm8k", "EleutherAI/hendrycks_math"
MODEL="mistralai/Mistral-Small-24B-Instruct-2501" # "HuggingFaceTB/SmolLM2-1.7B-Instruct", "meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen2.5-14B-Instruct", "mistralai/Mistral-Small-24B-Instruct-2501"
METHOD="top_p" # "greedy", "top_p", "best"
BATCH_SIZE=32 # 32, 64

MAX_NEW_TOKENS=512 # 512, 1024

TOP_P=0.9 # 0.9, 0.95
TEMPERATURE=0.8  # 0.6, 0.8

# 로그 모델 및 데이터 이름 설정
LOG_MODEL="mistral"
LOG_DATA="gsm"

# 3회 반복 실행
for RUN in {1..3}
do
  LOG_FILE="baseline_${LOG_MODEL}_${METHOD}_${LOG_DATA}_${MAX_NEW_TOKENS}tokens_run${RUN}.log"

  {
    echo "=== Inference Baselines: Run $RUN ==="
    echo "MODEL: $MODEL"
    echo "DATA: $DATA"
    echo "METHOD: $METHOD"
    echo "BATCH_SIZE: $BATCH_SIZE"
    echo "MAX_NEW_TOKENS: $MAX_NEW_TOKENS"
    echo "TOP_P: $TOP_P"
    echo "TEMPERATURE: $TEMPERATURE"
    echo "==============================="
  } > "./log/baseline/${LOG_MODEL}/${LOG_FILE}"

  echo ">>> Run ${RUN} 시작됨. 로그: ${LOG_FILE}"

  nohup python -u baseline_inference.py \
      --model "$MODEL" \
      --data "$DATA" \
      --method "$METHOD" \
      --batch_size $BATCH_SIZE \
      --max_new_tokens $MAX_NEW_TOKENS \
      --top_p $TOP_P \
      --temperature $TEMPERATURE \
      >> "./log/baseline/${LOG_MODEL}/${LOG_FILE}" 2>&1 &

  echo ">>> Run ${RUN} 종료됨."
done

