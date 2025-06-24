from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset
from dataset import MMLUDataset
from torch.utils.data import DataLoader



device = torch.device("cuda:5")


model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# MMLU 데이터셋 불러오기 (auxiliary_train 사용)
mmlu_train_dataset = load_dataset("cais/mmlu", "all")['auxiliary_train']

# custom dataset 래퍼 (만약 내부 데이터 수정이 가능하다면 해당 로직을 활용할 수 있습니다)
train_dataset = MMLUDataset(mmlu_train_dataset, tokenizer)

# 평가를 위한 DataLoader 생성 (순서를 보존하기 위해 shuffle=False)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)

# 모델 로드 (FP16 최적화)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)
model = model.to(device)
model.eval()

# 평가 및 labeling 변수 초기화
total_count = 0
correct_count = 0
# 각 샘플에 대한 예측 정답 여부를 저장하는 리스트 (나중에 Hugging Face 데이터셋에 컬럼으로 추가)
target_list = []

for batch in train_dataloader:
    total_count += 1

    # input_ids, attention_mask, prompt 및 정답 추출
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    prompt = batch["prompt"][0]
    ground_truth = batch["correct_answer"][0].strip()

    # batch가 1차원일 경우 차원 추가
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(0)

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    # 정답 문자열의 토큰 길이 확인
    ground_truth_token_ids = tokenizer.encode(ground_truth, add_special_tokens=False)
    ground_truth_len = len(ground_truth_token_ids)

    prompt_length = input_ids.shape[1]
    
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
    
    # 생성된 전체 텍스트 (prompt 포함)
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    # prompt 이후, 정답 토큰 길이만큼 토큰 추출
    generated_after_prompt = generated_ids[0][prompt_length:]
    predicted_token_ids = generated_after_prompt[:ground_truth_len]
    predicted_answer = tokenizer.decode(predicted_token_ids, skip_special_tokens=True).strip()
    
    # 정답 비교 (대소문자, 공백 차이 무시)
    is_correct = (predicted_answer.lower().strip() == ground_truth.lower().strip())
    if is_correct:
        correct_count += 1

    # 결과를 별도 리스트에 저장하는 대신, target_list에 1 (정답) 또는 0 (오답) 저장
    target_list.append(1 if is_correct else 0)

    # 진행 중 일부 결과 출력 (예: 처음 5개 샘플)
    if total_count <= 5:
        print(f"Sample {total_count}:")
        print("Prompt:")
        print(prompt)
        print("Ground Truth:", ground_truth)
        print("Predicted Answer:", predicted_answer)
        print("Is Correct:", is_correct)
        print("-" * 50)

# 전체 정확도 출력
accuracy = correct_count / total_count if total_count > 0 else 0
print(f"Total evaluated samples: {total_count}")
print(f"Correct predictions: {correct_count}")
print(f"Accuracy: {accuracy * 100:.2f}%")

# 평가 결과를 원래의 Hugging Face 데이터셋에 'target' 컬럼으로 추가
mmlu_train_dataset = mmlu_train_dataset.add_column("target", target_list)

# (선택사항) 추가된 target 컬럼을 확인하고 싶으면 다음과 같이 출력할 수 있습니다.
print(mmlu_train_dataset)
