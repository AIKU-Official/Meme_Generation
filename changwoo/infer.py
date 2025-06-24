import torch
from PIL import Image
from transformers import (
    VisionEncoderDecoderModel, 
    AutoTokenizer, 
    ViTImageProcessor
)
import os

## --- Configuration ---

# 1. Path to your fine-tuned model directory (should match OUTPUT_DIR from training)
MODEL_DIR = "vit-gpt2-oxford-hic-finetuned"

# 2. 🚨 *** IMPORTANT: CHANGE THIS to the path of the image you want to caption ***
IMAGE_PATH = "/home/aikusrv02/meme/Oxford_HIC/data/hic_data/images/bokete_1.jpg" 

# 3. Set device (use GPU if available, otherwise CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 4. Generation parameters (you can experiment with these)
MAX_LENGTH = 128  # Maximum length of the generated caption
NUM_BEAMS = 4     # Number of beams for beam search (higher can improve quality but is slower)

## --- Function to Generate Caption ---

def generate_caption(model, image_processor, tokenizer, image_path, device):
    """
    Loads an image, processes it through the model, and returns the generated caption.
    """
    try:
        # Load the image using PIL (Python Imaging Library)
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        return f"❌ Error: Image not found at '{image_path}'. Please check the path."
    except Exception as e:
        return f"❌ Error loading image: {e}"

    try:
        # Process the image: resize, normalize, and convert to tensor
        pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
        # Move the processed image tensor to the selected device (GPU/CPU)
        pixel_values = pixel_values.to(device)

        # Generate token IDs using the model's generate method
        # We use torch.no_grad() to disable gradient calculations - it's faster for inference.
        with torch.no_grad():
            output_ids = model.generate(
                pixel_values, 
                max_length=MAX_LENGTH, 
                num_beams=NUM_BEAMS,
                early_stopping=True # Stop generation when an end-of-sequence token is found
            )

        # Decode the generated token IDs back into text
        # We skip special tokens (like <BOS>, <EOS>, <PAD>) for a cleaner output.
        captions = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        
        # The output is a list, even for a single image. We return the first caption, stripped of whitespace.
        return captions[0].strip()

    except Exception as e:
        return f"❌ Error during caption generation: {e}"


## --- Main Inference Script ---

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    # 1. Check if the model directory exists
    if not os.path.isdir(MODEL_DIR):
        print(f"❌ Error: Model directory not found at '{MODEL_DIR}'.")
        print("   Did you run the training script and was it saved correctly?")
        exit()

    # 2. Load the fine-tuned model, tokenizer, and image processor
    print(f"Loading model from: {MODEL_DIR}")
    try:
        model = VisionEncoderDecoderModel.from_pretrained(MODEL_DIR)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        image_processor = ViTImageProcessor.from_pretrained(MODEL_DIR)
        
        # Move the model to the selected device
        model.to(DEVICE)
        # Set the model to evaluation mode (disables dropout, etc.)
        model.eval() 
        print("✅ Model, tokenizer, and image processor loaded successfully.")
        
    except Exception as e:
        print(f"❌ Error loading the model or its components: {e}")
        exit()

    # 3. Check if the user has set the IMAGE_PATH
    if IMAGE_PATH == "path/to/your/image.jpg":
        print("\n" + "="*60)
        print("🚨 Please edit the script and change the `IMAGE_PATH` variable")
        print("   to point to an image file you want to caption.")
        print("="*60)
        exit()

    # 4. Generate the caption
    print(f"\nGenerating caption for: {IMAGE_PATH}")
    caption_text = generate_caption(model, image_processor, tokenizer, IMAGE_PATH, DEVICE)

    # 5. Print the result
    print("\n--- 📸 Generated Caption 📸 ---")
    print(f"   '{caption_text}'")
    print("-------------------------------")
    