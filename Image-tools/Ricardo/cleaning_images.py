import os
import shutil
import time
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import numpy as np
from tqdm import tqdm

# Load a pre-trained model and processor from Hugging Face
model_name = "facebook/detr-resnet-50"
processor = DetrImageProcessor.from_pretrained(model_name)
model = DetrForObjectDetection.from_pretrained(model_name)

# Paths
folder_path = r"C:\Finishing_scrapes\Rolex\white_background_images"
output_folder = r"C:\Finishing_scrapes\Rolex\cleaned_images"


# Create output folders if they don't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Function to load images and preprocess for CUDA/CPU
def load_images(image_paths, processor, device):
    images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
    inputs = processor(images=images, return_tensors="pt").to(device)
    return inputs

# Function to process images in batches and measure time
def process_images(device, output_folder, batch_size=12):
    model.to(device)
    start_time = time.time()
    image_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, filename))]

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing batches"):
        batch_paths = image_paths[i:i + batch_size]

        # Load and preprocess the images
        inputs = load_images(batch_paths, processor, device)

        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)

        # Get the bounding boxes, labels, and scores
        target_sizes = torch.tensor([inputs['pixel_values'].shape[2:]] * len(batch_paths)).to(device)
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes)

        # Define a threshold for detection
        threshold = 0.5

        for idx, image_path in enumerate(batch_paths):
            watch_detected = False
            for score, label, box in zip(results[idx]["scores"], results[idx]["labels"], results[idx]["boxes"]):
                if score > threshold and label == 85:  # 'clock' label in COCO dataset
                    watch_detected = True
                    break

            if watch_detected:
                # Copy the watch image to the output folder
                shutil.copy(image_path, os.path.join(output_folder, os.path.basename(image_path)))
                print(f"Detected watch in image: {os.path.basename(image_path)}")
            else:
                print(f"No watch detected in image: {os.path.basename(image_path)}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    return elapsed_time

# Process images with GPU
gpu_time = process_images(torch.device('cuda'), output_folder)

# Process images with CPU
#cpu_time = process_images(torch.device('cpu'), cpu_output_folder)

# Print the processing times
print(f"Processing time with GPU: {gpu_time:.2f} seconds")
#print(f"Processing time with CPU: {cpu_time:.2f seconds")

print("Watch image separation completed.")
