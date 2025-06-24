import pandas as pd
from datasets import Dataset, Image as HFImage
from transformers import (
    AutoTokenizer, 
    ViTImageProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    default_data_collator,
)
import torch
from PIL import Image as PILImage
import os

os.environ["WANDB_DISABLED"] = "true"

## --- Configuration ---
# ⚠️ **Important:** Adjust these paths and settings according to your environment.

# 1. Set the base directory for your dataset
DATASET_BASE_DIR = "Oxford_HIC" 

# 2. Define paths based on the base directory
DATA_DIR = os.path.join(DATASET_BASE_DIR, "data", "hic_data")
CSV_PATH = os.path.join(DATA_DIR, "oxford_hic_data.csv")
IMAGE_DIR = os.path.join(DATA_DIR, "images")

# 3. Specify the image file extension (e.g., .jpg, .png)
# 🚨 *** YOU MUST VERIFY THIS *** - Check your 'images' folder.
IMAGE_EXTENSION = ".jpg" 

# 4. Set the pre-trained model ID
MODEL_ID = "nlpconnect/vit-gpt2-image-captioning"

# 5. Define the output directory for the fine-tuned model
OUTPUT_DIR = "vit-gpt2-oxford-hic-finetuned"

# 6. Training Hyperparameters (Adjust based on your hardware)
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
NUM_EPOCHS = 3
LEARNING_RATE = 5e-5
EVAL_STEPS = 500
LOGGING_STEPS = 100
SAVE_STEPS = 1000
MAX_CAPTION_LENGTH = 128 # Maximum length for tokenized captions
USE_FP16 = torch.cuda.is_available() # Use mixed precision if a GPU is available


## --- 1. Load and Prepare the Dataset ---

print(f"Loading CSV from: {CSV_PATH}")
try:
    df = pd.read_csv(CSV_PATH)
    print(f"Successfully loaded {len(df)} rows.")
except FileNotFoundError:
    print(f"❌ Error: CSV file not found at {CSV_PATH}. Please check the path.")
    exit()

# Drop rows with no captions
df = df.dropna(subset=['caption'])
print(f"Using {len(df)} rows after dropping NaNs.")

# Create a full image path column
def create_image_path(image_id):
    return os.path.join(IMAGE_DIR, f"{image_id}{IMAGE_EXTENSION}")

df['image_path'] = df['image_id'].apply(create_image_path)

# --- NEW: Filter out rows with missing images ---
initial_rows = len(df)
print(f"Checking for existing images (initial rows: {initial_rows})... This might take a moment.")
# Use os.path.exists to check if each file exists
df['image_exists'] = df['image_path'].apply(os.path.exists) 
# Keep only rows where the image exists
df_filtered = df[df['image_exists']].copy() # Use .copy() to avoid SettingWithCopyWarning
final_rows = len(df_filtered)

if initial_rows > final_rows:
    print(f"⚠️  Removed {initial_rows - final_rows} rows due to missing image files.")
else:
    print("✅ All image files listed in the CSV seem to exist.")

# We no longer need the 'image_exists' column
df_filtered = df_filtered.drop(columns=['image_exists'])
# --- End of NEW section ---


# Create a Hugging Face Dataset from the *filtered* DataFrame
print(f"Creating Hugging Face dataset with {len(df_filtered)} valid entries.")
dataset_dict = {"image": df_filtered["image_path"].tolist(), "text": df_filtered["caption"].tolist()}
dataset = Dataset.from_dict(dataset_dict).cast_column("image", HFImage())

# Split into training and validation sets (90% train, 10% test)
dataset = dataset.train_test_split(test_size=0.1)
train_ds = dataset["train"]
test_ds = dataset["test"]

print(f"Train dataset size: {len(train_ds)}")
print(f"Test dataset size: {len(test_ds)}")

# Check if dataset is empty after filtering
if len(train_ds) == 0 or len(test_ds) == 0:
    print("❌ Error: Not enough data left after filtering. Please check your image paths and CSV file.")
    exit()

## --- 2. Load Processor and Model ---

print(f"Loading tokenizer, image processor, and model: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
image_processor = ViTImageProcessor.from_pretrained(MODEL_ID)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID)

# Set decoder tokens
tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
model.config.eos_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id 
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.vocab_size = model.config.decoder.vocab_size


## --- 3. Preprocessing Function ---

def preprocess_data(examples):
    """
    Preprocesses images and captions for the model.
    """
    try:
        images = [img.convert("RGB") for img in examples["image"]]
        pixel_values = image_processor(images, return_tensors="pt").pixel_values
        
        labels = tokenizer(
            examples["text"],
            padding="max_length",
            max_length=MAX_CAPTION_LENGTH,
            truncation=True,
        ).input_ids

        labels = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label]
            for label in labels
        ]
        
        return {"pixel_values": pixel_values, "labels": labels}
    except Exception as e:
        # This catch is a fallback, filtering should prevent most errors
        print(f"Error processing batch: {e}. Skipping batch elements.")
        # Return empty dict or handle specific examples if needed
        return {}


train_ds.set_transform(preprocess_data)
test_ds.set_transform(preprocess_data)
print("Dataset preprocessing configured.")


## --- 4. Define Training Arguments ---

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    predict_with_generate=True,
    evaluation_strategy="steps",
    eval_steps=EVAL_STEPS,
    logging_steps=LOGGING_STEPS,
    save_steps=SAVE_STEPS,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    save_total_limit=2,         
    fp16=USE_FP16,              
    push_to_hub=False,          
    dataloader_num_workers=2,   
    # report_to="tensorboard",    
    remove_unused_columns=False
)
print("Training arguments defined.")


## --- 5. Initialize the Trainer ---

trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer, 
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    data_collator=default_data_collator, 
)
print("Trainer initialized. Starting training... 🚀")


## --- 6. Start Fine-Tuning ---

try:
    trainer.train()
    print("✅ Training completed successfully!")

    print("Saving final model, tokenizer, and image processor...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    image_processor.save_pretrained(OUTPUT_DIR)
    print(f"Model and components saved to {OUTPUT_DIR}")

except Exception as e:
    print(f"❌ An error occurred during training: {e}")
