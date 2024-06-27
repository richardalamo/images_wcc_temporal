from PIL import Image
import os
from tqdm import tqdm

def crop_image(image):
    left = 640
    upper = 240
    right = 1245
    lower = 850
    # Crop the image
    cropped_img = image.crop((left, upper, right, lower))
    return cropped_img

def process_images(input_folder, output_folder):
   
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png', '.jpeg', '.avif'))]

    for filename in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(input_folder, filename)
        image = Image.open(image_path).convert('RGBA')
        image= crop_image(image)       
        output_path = os.path.join(output_folder, filename)
        image.save(output_path)           
        

if __name__ == "__main__":    
    input_folder = r'C:\Users\Richard\Downloads\IWC_original'
    output_folder = r'C:\Users\Richard\Downloads\IWC_original_2'
    process_images(input_folder, output_folder)