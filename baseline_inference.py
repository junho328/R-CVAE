from datasets import load_dataset, concatenate_datasets
import math
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.sampling_params import BeamSearchParams

import os
import argparse
import re

# qwen_system_prompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant that solves math problems step-by-step with clear reasoning."

# instruction = """
# Your task:
# 1. Break down the solution into clear, logical steps.
# 2. The last line of your response should be of the form "####Answer: $ANSWER" where $ANSWER is the answer to the problem.

# **Question**:
# {question}

# **Answer**:
# """

instruction = """
You are a helpful assistant that solves math problems step-by-step with clear reasoning.
Your task:
1. Break down the solution into clear, logical steps.
2. The last line of your response should be of the form "####Answer: $ANSWER" where $ANSWER is the answer to the problem.

**Question**:
{question}

**Answer**:
"""

# instruction = '''
# You are a helpful assistant that solves math problems step-by-step with clear reasoning.
# Your task:
# 1. Break down the solution into clear, logical steps.
# 2. Always include a "Final answer:" section.
# 3. The **final answer** must be written strictly on the line **immediately following** "Final answer:" and must start with "####".
# 4. After "####", write **only the final result** without any extra words, symbols, or punctuation.
# 5. Absolutely do not include any additional text or commentary outside these steps or in the final answer.

# Here are examples:
# # Example 1
# **Question:**
# Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?

# **Answer:**
# Natalia sold 48/2 = <<48/2=24>>24 clips in May. 
# Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May

# **Final answer:
# ####72**

# # Example 2
# **Question:**
# On a particular day in Salt Lake, UT, the temperature was given by $-t^2 +12t+50$ where $t$ is the time in hours past noon. What is the largest $t$ value at which the temperature was exactly 77 degrees?

# **Answer:**
# We set the temperature equal to 77 degrees: \begin{align*}
# -t^2 +12t+50&=77\\
# t^2-12t+27&=0\\
# (t-3)(t-9)&=0
# \end{align*}
# We see then that the temperature is 77 degrees exactly twice: at $t=3$ and $t=9$, so our answer is $\boxed{9}$.
# x
# **Final answer:
# ####9**

# Now, solve the following problem:
# **Question:**
# '''

from math_verify import verify,parse

def extract_verify_answer(pred, label):
    try:
        pred = pred.split("####")[-1]
    except:
        print("Split error")
        return False
    
    print(f"Model Generated Answer : {pred}")
    print(f"Gold Label Answer : {label}")

    try:
        gold = parse(label)
    except:
        print("Gold parse error")
        return False

    try:
        answer = parse(pred)
    except:
        print("Answer parse error")
        return False
    
    print(f"Model Answer Parse: {answer}")
    print(f"Gold Answer Parse : {gold}")

    # Order here is important!
    return verify(gold, answer)

def main(args):

    if args.data == "openai/gsm8k":

        test_dataset = load_dataset("openai/gsm8k", "main", split="test")

    elif args.data == "EleutherAI/hendrycks_math":

        dataset1 = load_dataset("EleutherAI/hendrycks_math","algebra",split="test")
        dataset2 = load_dataset("EleutherAI/hendrycks_math","counting_and_probability",split="test")
        dataset3 = load_dataset("EleutherAI/hendrycks_math","geometry",split="test")
        dataset4 = load_dataset("EleutherAI/hendrycks_math","intermediate_algebra",split="test")
        dataset5 = load_dataset("EleutherAI/hendrycks_math","number_theory",split="test")
        dataset6 = load_dataset("EleutherAI/hendrycks_math","prealgebra",split="test")
        dataset7 = load_dataset("EleutherAI/hendrycks_math","precalculus",split="test")

        test_dataset = concatenate_datasets([dataset1, dataset2, dataset3, dataset4, dataset5, dataset6, dataset7]).shuffle()

    elif args.data == "HuggingFaceH4/MATH-500":

        test_dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")

    batch_size = args.batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = args.model
    llm = LLM(model_name, 
              device=device, 
              max_model_len=4096, 
              dtype="bfloat16",
              gpu_memory_utilization=0.7)

    print("\n>>> Models loaded!\n")

    all_generated_answers = []
    all_expected_answers = []

    num_batches = math.ceil(len(test_dataset) / batch_size)

    # 4. 배치 단위로 처리
    for i in tqdm(range(num_batches)):
        # 문제와 정답 배치 추출

        if args.data == "openai/gsm8k":

            questions = test_dataset["question"][i * batch_size : (i + 1) * batch_size]
            batch_expected = test_dataset["answer"][i * batch_size : (i + 1) * batch_size]

        elif args.data == "EleutherAI/hendrycks_math":

            questions = test_dataset["problem"][i * batch_size : (i + 1) * batch_size]
            batch_expected = test_dataset["solution"][i * batch_size : (i + 1) * batch_size]

        elif args.data == "HuggingFaceH4/MATH-500":

            questions = test_dataset["problem"][i * batch_size : (i + 1) * batch_size]
            batch_expected = test_dataset["answer"][i * batch_size : (i + 1) * batch_size]

        if args.method == "greedy":

            # vllm 샘플링 파라미터 설정
            params = SamplingParams(
                        max_tokens=args.max_new_tokens,
                        temperature=1.0,
                    )
            
        elif args.method == "beam":

            params = BeamSearchParams(beam_width=args.beam_width,
                                    max_tokens=args.max_new_tokens)

        elif args.method == "top_p": # Top-p sampling
            
            params = SamplingParams(
                        max_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                    )
            
        # chat_prompts = []
        # for q in questions:
        #     messages = [
        #         {"role": "system", "content": qwen_system_prompt},
        #         {"role": "user", "content": instruction.format(question=q)}
        #     ]
        #     # apply_chat_template을 이용해 chat 템플릿 적용 (문자열 생성)
        #     chat_text = tokenizer.apply_chat_template(
        #         messages,
        #         tokenize=False,
        #         add_generation_prompt=True
        #     )
        #     chat_prompts.append(chat_text)

        # outputs = llm.generate(chat_prompts, params)
        
        # vllm을 이용하여 배치 단위 문장 생성
        # 프롬프트 생성 (예: instruction 템플릿에 문제 삽입)
        if args.method == "beam":
            prompt_batch = [{"prompt": instruction.format(question=q)} for q in questions]
            outputs = llm.beam_search(prompt_batch, params)
            batch_generated = [output.sequences[0].text for output in outputs]
        else:    
            prompt_batch = [instruction.format(question=q) for q in questions]
            outputs = llm.generate(prompt_batch, params)
            batch_generated = [output.outputs[0].text for output in outputs]
        
        all_generated_answers.extend(batch_generated)
        all_expected_answers.extend(batch_expected)

        del outputs, prompt_batch, batch_generated, batch_expected

    if args.data == "openai/gsm8k":
        all_expected_answers = [item.split("####")[-1].strip() for item in all_expected_answers]

    elif args.data == "EleutherAI/hendrycks_math":
        pattern = r'\\boxed\{([^}]*)\}'
        all_expected_answers = [re.search(pattern, answer).group(1) for answer in all_expected_answers]

    correct = 0
    for idx,(gen,exp) in enumerate(zip(all_generated_answers, all_expected_answers)):
        print(">>> Verifying {idx} answer\n")
        result = extract_verify_answer(gen,exp)
        correct += result

    print(f"Accuracy: {correct / len(all_expected_answers) * 100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--data", type=str, default="openai/gsm8k")
    parser.add_argument("--method", type=str, default="greedy")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--beam_width", type=int, default=8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    main(args)

