# Install dependencies:
# pip install torch transformers datasets accelerate pillow

import os
import pandas as pd
from datasets import Dataset, DatasetDict
from PIL import Image
import torch
from transformers import (
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)

os.environ["WANDB_DISABLED"] = "true"

def load_and_split(csv_path: str, images_folder: str, split_ratio: float = 0.9, seed: int = 42) -> DatasetDict:
    # Read CSV with explicit caption dtype to avoid mixed-type warning
    df = pd.read_csv(csv_path, dtype={'caption': str}, low_memory=False)
    df['caption'] = df['caption'].astype(str)
    df['image_path'] = df['image_id'].apply(lambda x: os.path.join(images_folder, f"{x}.jpg"))
    df = df[df['image_path'].apply(os.path.exists)].reset_index(drop=True)
    train_df = df.sample(frac=split_ratio, random_state=seed)
    val_df = df.drop(train_df.index)
    return DatasetDict({
        'train': Dataset.from_pandas(train_df, preserve_index=False),
        'validation': Dataset.from_pandas(val_df, preserve_index=False)
    })


def main():
    model_name = "nlpconnect/vit-gpt2-image-captioning"
    csv_path = "/home/aikusrv02/meme/Oxford_HIC/data/hic_data/oxford_hic_data.csv"
    images_folder = "/home/aikusrv02/meme/Oxford_HIC/data/hic_data/images"

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    datasets = load_and_split(csv_path, images_folder)

    # Load model and processors
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    model.to(device)
    feature_extractor = ViTImageProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Generation settings
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.max_length = 128
    model.config.num_beams = 4

    # Prepare transform on the fly
    def preprocess_fn(example):
        # Handle list vs single-value for image_path
        img_path = example['image_path']
        if isinstance(img_path, (list, tuple)):
            img_path = img_path[0]
        img = Image.open(img_path).convert("RGB")
        pixel_values = feature_extractor(images=img, return_tensors="pt").pixel_values
        # Handle list vs single-value for caption
        caption = example['caption']
        if isinstance(caption, (list, tuple)):
            caption = caption[0]
        tokenized = tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=model.config.max_length
        )
        labels = [ (token if token != tokenizer.pad_token_id else -100) for token in tokenized.input_ids ]
        return {"pixel_values": pixel_values, "labels": labels}

    train_dataset = datasets['train'].with_transform(preprocess_fn)
    eval_dataset  = datasets['validation'].with_transform(preprocess_fn)
    print(train_dataset[0])

    # Collate into batches
    def collate_fn(batch):
        # pixel_values = torch.stack([ex['pixel_values'] for ex in batch])
        # if len(batch) == 1:
        #     # If batch size is 1, we need to unsqueeze to maintain the expected shape
        #     pixel_values = batch[0]['pixel_values'].unsqueeze(0)
        #     labels       = torch.tensor(batch[0]['labels'], dtype=torch.long).unsqueeze(0)
        # else:
        # Concatenate pixel values from all examples in the batch
        pixel_values = torch.cat([ex['pixel_values'] for ex in batch], dim=0)
        labels       = torch.tensor([ex['labels'] for ex in batch], dtype=torch.long)

        print("shape:", pixel_values.shape)  # Should be [batch_size, channels, height, width]
        return {"pixel_values": pixel_values, "labels": labels}

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="./vit-gpt2-finetuned",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        evaluation_strategy="steps",
        num_train_epochs=3,
        fp16=torch.cuda.is_available(),
        logging_steps=100,
        save_steps=500,
        eval_steps=500,
        save_total_limit=2,
        remove_unused_columns=False
    )

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        tokenizer=tokenizer
    )

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
