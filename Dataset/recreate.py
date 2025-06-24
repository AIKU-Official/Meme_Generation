import os
import pandas as pd
import torch
import pickle
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

# 경로 설정
csv_path = '/home/aikusrv02/meme/minyoung/CLIP_prefix_caption/data/Oxford_HIC/oxford_hic_data.csv'
img_dir = '/home/aikusrv02/meme/minyoung/CLIP_prefix_caption/data/Oxford_HIC/hic_data/images'
out_pkl = '/home/aikusrv02/meme/minyoung/CLIP_prefix_caption/data/Oxford_HIC/train_clipcap.pkl'

# CLIP 준비 (ViT-B/32 예시)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval().cuda()
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 데이터프레임 로드
df = pd.read_csv(csv_path)

# image_id → image_path 추가
df["image_path"] = df["image_id"].apply(lambda x: os.path.join(img_dir, f"{x}.jpg"))
df = df[df["image_path"].apply(os.path.exists)].reset_index(drop=True)

# image_id별로 CLIP 임베딩 추출 (중복 제거)
unique_image_ids = df["image_id"].unique()
imageid2idx = {img_id: idx for idx, img_id in enumerate(unique_image_ids)}
clip_embeddings = []

for img_id in tqdm(unique_image_ids):
    img_path = os.path.join(img_dir, f"{img_id}.jpg")
    image = Image.open(img_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt").to("cuda")
    with torch.no_grad():
        img_feat = clip_model.get_image_features(**inputs)
    clip_embeddings.append(img_feat.cpu())

clip_embeddings = torch.cat(clip_embeddings, dim=0)  # [N, D]

# 캡션-임베딩 매핑 (중복 캡션 허용)
captions = []
for _, row in df.iterrows():
    captions.append({
        "caption": row["caption"],
        "image_id": row["image_id"],
        "clip_embedding": imageid2idx[row["image_id"]]
    })

# 저장
data = {
    "clip_embedding": clip_embeddings,  # [N, D]
    "captions": captions                # [{caption, image_id, clip_embedding}]
}
with open(out_pkl, "wb") as f:
    pickle.dump(data, f)
print(f"Dataset saved to {out_pkl} with {len(captions)} captions and {clip_embeddings.shape[0]} image embeddings.")
