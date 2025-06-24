import os
import pandas as pd
from PIL import Image

# Hugging Face libraries
from datasets import Dataset, DatasetDict
from transformers import (
    ViTFeatureExtractor,
    GPT2Tokenizer,
    VisionEncoderDecoderModel,
    TrainingArguments,
    Trainer
)

def main():
    # 1) LOAD THE CSV AND PREPARE A HUGGING FACE DATASET
    csv_path = "/home/aikusrv02/meme/Oxford_HIC/data/hic_data/oxford_hic_data.csv"
    image_dir = "/home/aikusrv02/meme/Oxford_HIC/data/hic_data/images"

    # Load CSV into a pandas DataFrame
    df = pd.read_csv(csv_path)

    # Add an 'image_path' column by appending file extension (e.g., '.jpg')
    # Adjust extension as needed (png, jpeg, etc.).
    # If your images have different naming, modify accordingly.
    def make_image_path(row):
        image_id = row["image_id"]
        return os.path.join(image_dir, f"{image_id}.jpg")

    df["image_path"] = df.apply(make_image_path, axis=1)

    # Create a Hugging Face Dataset from the DataFrame
    dataset_hf = Dataset.from_pandas(df)

    # OPTIONAL: If you do not have separate CSVs for train and val, split here:
    dataset_dict = dataset_hf.train_test_split(test_size=0.1, seed=42)
    train_ds = dataset_dict["train"]
    val_ds = dataset_dict["test"]

    # 2) INITIALIZE FEATURE EXTRACTOR & TOKENIZER
    feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = GPT2Tokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    # GPT-2 has no default pad token, so set pad_token to eos_token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3) PREPROCESS FUNCTION FOR EACH EXAMPLE
    def preprocess(example, max_caption_length=50):
        # Load the image from the local path
        image = Image.open(example["image_path"]).convert("RGB")

        # Extract pixel values
        pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values[0]

        # Tokenize caption
        caption_tokens = tokenizer(
            example["caption"],
            truncation=True,
            padding="max_length",
            max_length=max_caption_length,
            return_tensors="pt"
        )["input_ids"][0]

        # Return the preprocessed features
        return {
            "pixel_values": pixel_values,
            "labels": caption_tokens,
        }

    # 4) APPLY PREPROCESSING TO DATASETS
    def transform_dataset(ds):
        # Map the preprocessing, but keep batched=False to avoid nested structures
        ds = ds.map(preprocess, batched=False)

        # Remove original columns to keep dataset clean (optional)
        # You can skip this if you want to keep 'image_id', 'caption', etc.
        ds = ds.remove_columns(["image_id", "caption", "funny_score", "image_path"])
        return ds

    train_ds = transform_dataset(train_ds)
    val_ds = transform_dataset(val_ds)

    # 5) LOAD THE PRETRAINED MODEL
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    # Update special tokens / config
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    # Set generation hyperparameters (for inference)
    model.config.max_length = 50
    model.config.num_beams = 4

    # 6) TRAINING SETUP
    training_args = TrainingArguments(
        output_dir="./oxford_hic-finetuned",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        learning_rate=5e-5,
        evaluation_strategy="steps",
        save_steps=100,
        eval_steps=100,
        logging_steps=50,
        predict_with_generate=True,     # For image captioning tasks
        fp16=True,                      # Enable mixed precision
    )

    # We don’t strictly need a custom collator for vision-text data if we’ve handled everything in `preprocess`.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    # checkpoints will be stored in `./oxford_hic-finetuned`.
    trainer.train()


if __name__ == "__main__":
    main()
