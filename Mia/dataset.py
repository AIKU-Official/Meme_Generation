from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, random_split


def train_test_split(dataset, test_size=0.2, seed=None):
    N = len(dataset)
    N_test = int(test_size * N)
    N -= N_test

    if seed is not None:
        train, test = random_split(
            dataset, [N, N_test], generator=torch.Generator().manual_seed(seed)
        )
    else:
        train, test = random_split(dataset, [N, N_test])

    return train, test



def get_loader(dataset, batch_size=128, accelerator=None):
    loader = DataLoader(
        dataset, batch_size=batch_size)
    if accelerator is not None:
        loader = accelerator.prepare(loader)
    return loader


# 사용자 정의 데이터셋 클래스
class MMLUDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        question = sample['question']
        subject = sample['subject']
        choices = sample['choices']
        answer_index = sample['answer']
        correct_answer = choices[answer_index]  # 실제 정답
        
        LLAMA_3_SYS_PROMPT = "You are an expert who responds with concise, correct answers. Directly state the answer without phrases like 'the correct answer is'"

        # 프롬프트 구성
        prompt = (
            f"You are an expert who responds with concise, correct answers. Directly state the answer among choices without phrases like 'the correct answer is'.\n"
            f"[Question]: {question}\n"
            f"[Choices]: {', '.join(choices)}\n"
            f"[Answer]: "
        )

        encoding = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "choices": choices,
            "correct_answer": correct_answer,
            "answer_index": answer_index,
            "prompt": prompt
        }
    
