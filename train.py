from sentence_transformers import SentenceTransformer
from datasets import load_dataset, concatenate_datasets
import numpy as np
import re
import regex
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from model import Encoder, Decoder, ReasoningPolicy

import argparse
import os

instruction = """
You are a helpful assistant that solves math problems step-by-step with clear reasoning.
Your task:
1. Break down the solution into clear, logical steps.
2. The last line of your response should be of the form "####Answer: $ANSWER" where $ANSWER is the answer to the problem.

**Question**:
{question}

**Answer**:
"""

def preprocess_gsm(example):
    answer = example["answer"].split("####")[-1].strip()
    example["solution"] = example["answer"].split("####")[0].strip()
    return example

# reparameterization trick: 주어진 평균과 로그 분산으로부터 latent z 샘플 생성
def reparameterize(mean, logvar):
    std = torch.exp(0.5 * logvar)  # 표준편차 계산
    eps = torch.randn_like(std)    # 정규분포로부터 잡음 샘플링
    return mean + eps * std

# KL divergence를 계산하는 함수
def kl_divergence(mean, logvar):
    # 정규분포 N(0, I)를 prior로 가정했을 때, KL divergence 공식:
    # D_KL = -0.5 * sum(1 + logvar - mean^2 - exp(logvar))
    kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1)
    return torch.mean(kl)

def kl_divergence_two_gaussians(mu_policy, logvar_policy, mu_target, logvar_target):
    """
    KL(N(policy) || N(target)) = 0.5 * sum[ log(sigma_target^2/sigma_policy^2) + 
        (sigma_policy^2 + (mu_policy - mu_target)^2)/sigma_target^2 - 1 ]
    """
    # sigma^2 = exp(logvar)
    kld_element = 0.5 * (logvar_target - logvar_policy + 
                        (torch.exp(logvar_policy) + (mu_policy - mu_target).pow(2)) / torch.exp(logvar_target) - 1)
    # 배치별로 latent_dim에 대해 합산한 뒤 평균을 취함
    return torch.mean(torch.sum(kld_element, dim=1))

def train_cvae_step(encoder, decoder, optimizer_cvae, batch_q, batch_r):
    optimizer_cvae.zero_grad()
    
    # Encoder: (Q, R) -> latent 분포 파라미터 및 샘플 z_enc 생성
    _, enc_mean, enc_logvar = encoder(batch_q, batch_r)
    z_enc = reparameterize(enc_mean, enc_logvar)
    
    # Decoder: (Q, z_enc) -> R 임베딩 재구성
    r_pred = decoder(batch_q, z_enc)
    
    # Reconstruction loss (MSE)와 KL divergence 손실 계산
    recon_loss = F.mse_loss(r_pred, batch_r, reduction='mean')
    kl_loss = kl_divergence(enc_mean, enc_logvar)
    
    loss_cvae = recon_loss + args.kl_weight * kl_loss
    loss_cvae.backward()
    optimizer_cvae.step()
    
    return loss_cvae.item()

def train_policy_step(encoder, reasoning_policy, optimizer_policy, batch_q, batch_r):

    encoder.eval()
    
    if args.rationale_train_method == 'mse':

        with torch.no_grad():
            _, enc_mean, enc_logvar = encoder(batch_q, batch_r)
            z_enc = reparameterize(enc_mean, enc_logvar)
        encoder.train()  # 이후 다른 용도로 encoder가 필요한 경우 대비

        optimizer_policy.zero_grad()
        # Reasoning Policy: Q -> 예측 latent (policy_mean)
        _, policy_mean, _ = reasoning_policy(batch_q)
        
        # MSE 손실: encoder에서 생성한 z_enc (target)와 policy의 예측 값 비교
        rationale_loss = F.mse_loss(policy_mean, z_enc, reduction='mean')
        rationale_loss.backward()
        optimizer_policy.step()

    else:

        with torch.no_grad():
            _, target_mean, target_logvar = encoder(batch_q, batch_r)
        encoder.train()  # 이후 다른 용도로 encoder가 필요한 경우를 위해
        
        optimizer_policy.zero_grad()
        # Reasoning Policy: Q -> 예측 latent 분포 (policy_mean, policy_logvar)
        _, policy_mean, policy_logvar = reasoning_policy(batch_q)
        
        # KL divergence를 손실 함수로 사용하여 두 분포 간의 차이를 최소화
        rationale_loss = kl_divergence_two_gaussians(policy_mean, policy_logvar, target_mean, target_logvar)
        rationale_loss.backward()
        optimizer_policy.step()
        
    return rationale_loss.item()

def main(args):

    if args.load == False:

        # 1. Load and preprocess the data

        if args.data == "gsm_math":

            gsm_dataset = load_dataset("openai/gsm8k", "main", split="train")

            gsm_dataset = gsm_dataset.map(preprocess_gsm)
            gsm_Q = [instruction.format(question=question) for question in gsm_dataset["question"]]
            gsm_R = gsm_dataset["solution"]

            # Login using e.g. `huggingface-cli login` to access this dataset
            dataset1 = load_dataset("EleutherAI/hendrycks_math","algebra",split="train")
            dataset2 = load_dataset("EleutherAI/hendrycks_math","counting_and_probability",split="train")
            dataset3 = load_dataset("EleutherAI/hendrycks_math","geometry",split="train")
            dataset4 = load_dataset("EleutherAI/hendrycks_math","intermediate_algebra",split="train")
            dataset5 = load_dataset("EleutherAI/hendrycks_math","number_theory",split="train")
            dataset6 = load_dataset("EleutherAI/hendrycks_math","prealgebra",split="train")
            dataset7 = load_dataset("EleutherAI/hendrycks_math","precalculus",split="train")

            math_dataset = concatenate_datasets([dataset1, dataset2, dataset3, dataset4, dataset5, dataset6, dataset7, gsm_dataset]).shuffle()

            math_Q = [instruction.format(question=question) for question in math_dataset["problem"]]
            math_R = math_dataset["solution"]

            question = gsm_Q + math_Q
            rationale = gsm_R + math_R
        
        elif args.data == "numina":

            dataset = load_dataset("AI-MO/NuminaMath-CoT")
            
            question = [instruction.format(question=question) for question in dataset["problem"]]
            rationale = dataset["solution"]

        print(">>>Data preprocessing complete!")

        # 2. Embed data
        embed_model = SentenceTransformer(args.embed_model)
        
        question_embeddings = embed_model.encode(question)
        rationale_embeddings = embed_model.encode(rationale)

        print("# Question embeddings shape:", question_embeddings.shape)
        print("# Rationale embeddings shape:", rationale_embeddings.shape)

        embed_output = args.output + "/embeddings"
        os.makedirs(embed_output, exist_ok=True)

        np.save(embed_output + f"/{args.data}_question_embeddings.npy", question_embeddings)
        np.save(embed_output + f"/{args.data}_rationale_embeddings.npy", rationale_embeddings)

        print(">>>Embeddings saved successfully!")

    # 3. CVAE & Rationale Model Training

    embed_output = args.output + "/embeddings"
    question_embeddings = np.load(embed_output + f"/{args.data}_question_embeddings.npy")
    rationale_embeddings = np.load(embed_output + f"/{args.data}_rationale_embeddings.npy")
    
    print(">>>Embeddings loaded successfully!")

    q_dim = 384         # 질문 임베딩 차원
    r_dim = 384         # 근거 임베딩 차원
    latent_dim = args.z_dim         # 잠재 공간 차원
    r_emb_dim = 384     # Decoder 출력 차원

    encoder = Encoder(q_dim, r_dim, latent_dim)
    decoder = Decoder(q_dim, latent_dim, r_emb_dim)
    reasoning_policy = ReasoningPolicy(q_dim, latent_dim)

    optimizer_cvae = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.cvae_lr)
    optimizer_policy = optim.Adam(reasoning_policy.parameters(), lr=args.rationale_lr)

    # cosine learning rate scheduler 생성
    num_epochs_cvae = args.cvae_epochs
    num_epochs_policy = args.rationale_epochs

    scheduler_cvae = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_cvae, T_max=num_epochs_cvae)
    scheduler_policy = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_policy, T_max=num_epochs_policy)


    # ----------------------------
    # 더미 데이터 및 DataLoader 구성
    # ----------------------------
    dataset = TensorDataset(torch.tensor(question_embeddings), torch.tensor(rationale_embeddings))
    batch_size = args.batch_size
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # ----------------------------
    # Phase 1: CVAE (AutoEncoder) 학습
    # ----------------------------
    num_epochs_cvae = args.cvae_epochs
    print("=== Phase 1: CVAE Training ===")
    for epoch in range(num_epochs_cvae):
        total_loss = 0.0
        num_batches = 0
        for batch_q, batch_r in dataloader:
            loss = train_cvae_step(encoder, decoder, optimizer_cvae, batch_q, batch_r)
            total_loss += loss
            num_batches += 1
        avg_loss = total_loss / num_batches
        print(f'Epoch [{epoch+1}/{num_epochs_cvae}], CVAE Loss: {avg_loss:.4f}')
        scheduler_cvae.step()

    # ----------------------------
    # Phase 2: Reasoning Policy 학습 (AutoEncoder 훈련 후)
    # ----------------------------
    num_epochs_policy = args.rationale_epochs
    print("\n=== Phase 2: Reasoning Policy Training ===")
    for epoch in range(num_epochs_policy):
        total_loss = 0.0
        num_batches = 0
        for batch_q, batch_r in dataloader:
            loss = train_policy_step(encoder, reasoning_policy, optimizer_policy, batch_q, batch_r)
            total_loss += loss
            num_batches += 1
        avg_loss = total_loss / num_batches
        print(f'Epoch [{epoch+1}/{num_epochs_policy}], Policy Loss: {avg_loss:.4f}')
        scheduler_policy.step()

    print(">>>Training complete!")

    # 5. Save the model

    checkpoint_path = args.output + "/checkpoint/"
    
    # checkpoint 저장: 모델 state_dict와 옵티마이저 state_dict 포함
    checkpoint = {
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'reasoning_policy_state_dict': reasoning_policy.state_dict(),
        'optimizer_cvae_state_dict': optimizer_cvae.state_dict(),
        'optimizer_policy_state_dict': optimizer_policy.state_dict(),
        # 추가 정보 (예: 마지막 에포크, 손실값 등)도 포함 가능
        'epoch': num_epochs_cvae + num_epochs_policy
    }

    os.makedirs(checkpoint_path, exist_ok=True)
    torch.save(checkpoint, checkpoint_path + f'{args.rationale_train_method}_{args.kl_weight}_z{args.z_dim}_model_train.pth')

    print(">>>Model saved successfully!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a CVAE model')
    parser.add_argument('--data', type=str, default="gsm_math" ,help='training data')
    parser.add_argument('--load', type=bool, default=True, help='Load embeddings')
    parser.add_argument('--embed_model', type=str, default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument("--z_dim", type=int, default=256, help="Dimension of the latent space")
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--cvae_epochs', type=int, default=1000, help='Number of epochs to train the CVAE')
    parser.add_argument('--cvae_lr', type=float, default=1e-4, help='Learning rate for the CVAE')
    parser.add_argument('--kl_weight', type=float, default=4.0, help='KL divergence weight')
    parser.add_argument('--rationale_train_method', type=str, default='kld', help='MSE or KLD for training the Rationale')
    parser.add_argument('--rationale_epochs', type=int, default=1000, help='Number of epochs to train the Rationale')
    parser.add_argument('--rationale_lr', type=float, default=1e-4, help='Learning rate for the Rationale')
    parser.add_argument('--output', type=str, default="./output", help='Path to the output model')
    args = parser.parse_args()

    main(args)



