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
    Seq2SeqTrainer,
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


    # text preprocessing step
    def tokenization_fn(captions, max_target_length):
        """Run tokenization on captions."""
        labels = tokenizer(captions, 
                        padding="max_length", 
                        max_length=max_target_length).input_ids

        return labels

    # image preprocessing step
    def feature_extraction_fn(image_paths, check_image=True):
        """
        Run feature extraction on images
        If `check_image` is `True`, the examples that fails during `Image.open()` will be caught and discarded.
        Otherwise, an exception will be thrown.
        """

        model_inputs = {}

        if check_image:
            images = []
            to_keep = []
            for image_file in image_paths:
                try:
                    img = Image.open(image_file)
                    images.append(img)
                    to_keep.append(True)
                except Exception:
                    to_keep.append(False)
        else:
            images = [Image.open(image_file) for image_file in image_paths]

        encoder_inputs = feature_extractor(images=images, return_tensors="np")

        return encoder_inputs.pixel_values

    def preprocess_fn(examples, max_target_length, check_image = True):
        """Run tokenization + image feature extraction"""
        image_paths = examples['image_path']
        captions = examples['caption']    
        
        model_inputs = {}
        # This contains image path column
        model_inputs['labels'] = tokenization_fn(captions, max_target_length)
        model_inputs['pixel_values'] = feature_extraction_fn(image_paths, check_image=check_image)

        return model_inputs

    processed_dataset = datasets.map(
        function=preprocess_fn,
        batched=True,
        fn_kwargs={"max_target_length": 128},
        remove_columns=datasets['train'].column_names
    )
    print(processed_dataset)

    # Collate into batches
    def collate_fn(batch):
        # pixel_values = torch.stack([ex['pixel_values'] for ex in batch])
        print("batch", len(batch))
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
        train_dataset=processed_dataset['train'],
        eval_dataset=processed_dataset['validation'],
        data_collator=collate_fn,
        tokenizer=tokenizer
    )

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
