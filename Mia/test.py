from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
from io import BytesIO

# Load the BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=False)
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load an image
# Download an image from a URL
image_url = "https://i.pinimg.com/236x/34/ee/4d/34ee4d418a30e5ca3faf307386591fa7.jpg"  # Replace with the actual URL of your image
response = requests.get(image_url)
image = Image.open(BytesIO(response.content)).convert("RGB")
# The image is already loaded from the URL, so this line is removed.

# Prepare inputs
inputs = processor(image, return_tensors="pt")

# Generate caption
output = model.generate(**inputs)
caption = processor.decode(output[0], skip_special_tokens=True)

print("Generated Caption:", caption)