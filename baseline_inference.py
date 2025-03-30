from datasets import load_dataset, concatenate_datasets
import math
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.sampling_params import BeamSearchParams

import os
import argparse
import re

instruction = """
You are a helpful assistant that solves math problems step-by-step with clear reasoning.
Your task:
1. Break down the solution into clear, logical steps.
2. The last line of your response should be of the form "####Answer: $ANSWER" where $ANSWER is the answer to the problem.

**Question**:
{question}

**Answer**:
"""

from math_verify import verify,parse

def extract_verify_answer(pred, label):
    try:
        pred = pred.split("####")[-1]
    except:
        print("Split error")
        return False

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

    elif args.data == "Maxwell-Jia/AIME_2024":

        test_dataset = load_dataset("Maxwell-Jia/AIME_2024", split="train")

    batch_size = args.batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = args.model

    # if args.method == "best":

    #     llm = LLM(model_name, 
    #             device=device, 
    #             max_model_len=4096, 
    #             dtype="bfloat16",
    #             gpu_memory_utilization=0.7)
        
    # else:

    # if args.model == "mistralai/Mistral-Small-24B-Instruct-2501":
    #     llm = LLM(model="mistralai/Mistral-Small-24B-Instruct-2501",
    #               tokenizer_mode="mistral",
    #               config_format="mistral",
    #               load_format="mistral",
    #               dtype="bfloat16",
    #               device=device,
    #               max_model_len=4096)

    # else: 
    llm = LLM(model_name, 
        device=device, 
        max_model_len=4096, 
        dtype="bfloat16")

    print("\n>>> Models loaded!\n")

    if args.method == "best":
        
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        
        reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_model)
        reward_model = AutoModelForSequenceClassification.from_pretrained(args.reward_model).to(device)
        
        print("\n>>> Reward Model loaded!\n")

        if args.data == "openai/gsm8k":

            questions = test_dataset["question"]
            gold_answers =  [answer.split("####")[-1].strip() for answer in test_dataset["answer"]]

        elif args.data == "EleutherAI/hendrycks_math":

            questions = test_dataset["problem"]
            pattern = r'\\boxed\{([^}]*)\}'
            gold_answers = [re.search(pattern, solution).group(1) for solution in test_dataset["solution"]]

        elif args.data == "HuggingFaceH4/MATH-500":

            questions = test_dataset["problem"]
            gold_answers = test_dataset["answer"]

        elif args.data == "Maxwell-Jia/AIME_2024":

            questions = test_dataset["Problem"]
            gold_answers = test_dataset["Answer"]

        correct = 0

        for question,gold_answer in tqdm(zip(questions, gold_answers)):
            
            sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=args.max_new_tokens,
            n=args.num_samples  # ← 여기서 하나의 프롬프트에 대해 5개의 응답을 샘플링
            )

            llm_outputs = llm.generate([question], sampling_params)

            print("\n>>> Generating samples complete!\n")

            answer_samples = []

            for i,completion in enumerate(llm_outputs[0].outputs):
                answer_samples.append(completion.text)

            # 각 답변과 질문을 결합하여 입력 문자열 구성 (필요에 따라 형식을 조정)
            input_texts = [question + "\n" + answer for answer in answer_samples]
            inputs = reward_tokenizer(input_texts, return_tensors="pt", truncation=True, padding=True).to(device)

            with torch.no_grad():
                outputs = reward_model(**inputs)
                # 모델의 출력 형태에 따라 score 추출 (예: 로짓의 두 번째 클래스 혹은 단일 값)
                # 아래 예시는 로짓이 [batch_size, num_labels] 형태이고, 두 번째 클래스(logits[:, 1])가 긍정 점수라고 가정
                if outputs.logits.shape[-1] > 1:
                    scores = outputs.logits[:, 1]
                else:
                    scores = outputs.logits.squeeze(-1)
            best_sample_idx = int(torch.argmax(scores, dim=0).item())
            best_answer = answer_samples[best_sample_idx]

            result = extract_verify_answer(best_answer,gold_answer)
            correct += result

            del llm_outputs, outputs, inputs, scores
        
        print(f"Accuracy: {correct / len(questions) * 100:.2f}%")

    else:

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

            elif args.data == "Maxwell-Jia/AIME_2024":

                questions = test_dataset["Problem"][i * batch_size : (i + 1) * batch_size]
                batch_expected = test_dataset["Answer"][i * batch_size : (i + 1) * batch_size]

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

        print(f"Accuracy: {correct / len(all_expected_answers) * 100:.2f}%", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--data", type=str, default="openai/gsm8k")
    parser.add_argument("--method", type=str, default="greedy")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--beam_width", type=int, default=8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--reward_model", type=str, default="trl-lib/Qwen2-0.5B-Reward")
    args = parser.parse_args()

    main(args)

