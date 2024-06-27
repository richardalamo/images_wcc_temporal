import numpy as np
import os
from PIL import Image
from rembg import remove
import pillow_avif
from rembg import remove
from PIL import Image
from io import BytesIO
import os
import time
from tqdm import tqdm
import cv2
#installations needed
#pip install pillow-avif-plugin
#pip install rembg --user

def crop_transparent(image):
    """
    Crop the transparent area around an image.
    """

    # Convert to RGBA if the image is not
    if image.mode != 'RGBA':
        image = image.convert('RGBA')

    # Convert PIL image to OpenCV format
    open_cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGRA)

    # Threshold image to binary
    _, thresh = cv2.threshold(cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Find the bounding box of the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Add a buffer around the bounding box
    buffer = 10
    x = max(0, x - buffer)
    y = max(0, y - buffer)
    w = min(image.size[0], x + w + buffer) - x
    h = min(image.size[1], y + h + buffer) - y

    # Crop the image based on the bounding box
    image_cropped = image.crop((x, y, x + w, y + h))

    return image_cropped


def remove_background(input_image, file_extension):
    """
    Remove the background of the image using rembg for JPG images.
    For PNG images, return them as is.
    """
    if file_extension.lower() in ['.jpg', '.jpeg', '.avif','.png']:
        input_image = input_image.convert("RGBA")
        bytes_img = BytesIO()
        input_image.save(bytes_img, format="PNG")
        
        output_bytes = remove(bytes_img.getvalue())
        output_image = Image.open(BytesIO(output_bytes))
        
        return output_image
    elif file_extension.lower() == '.png':
        return input_image
    else:
        raise ValueError("Unsupported file extension")

def load_image(image_path):
    with open(image_path, "rb") as inp_file:
        return Image.open(BytesIO(inp_file.read()))



def resize_and_center_images(folder_path, output_folder, canvas_width=1000, canvas_height=1000, fixed_margin=10):
    """
    Resize and center images from the given folder and save to the output folder.
    
    Parameters:
        folder_path (str): The path to the folder containing the images.
        output_folder (str): The path to the folder where the processed images will be saved.
        canvas_width (int): The width of the canvas where the image will be pasted.
        canvas_height (int): The height of the canvas where the image will be pasted.
        min_margin (int): The minimum margin between the image and the canvas boundaries.
        
    Returns:
        None
    """
    
    # Create output folder if not exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg', '.avif'))]
    
    # Loop through the list of image files
    for filename in tqdm(image_files, desc="Processing images"):
        # Create the full path to the image file
        image_path = os.path.join(folder_path, filename)
    
        # Convert the read bytes to an image object
        with open(image_path, "rb") as inp_file:
            image = Image.open(BytesIO(inp_file.read()))
                
                               
        file_extension = os.path.splitext(filename)[1]

          
        #Removing background and croping
        # new_image =  remove_background(image, file_extension)
        new_image = crop_transparent(image)
        
        

        # Aspect ratio
        aspect_ratio = new_image.width / new_image.height
        
        max_width = canvas_width - 2 * fixed_margin
        max_height = canvas_height - 2 * fixed_margin

        # Calculate the scaling factor based on the smaller dimension
        scale_factor = min(max_width / new_image.width, max_height / new_image.height)
        
        # Calculate the scaling factor based on the smaller dimension
        scale_factor = min(max_width / new_image.width, max_height / new_image.height)

        # Apply the scaling factor to both dimensions
        new_width = int(new_image.width * scale_factor)
        new_height = int(new_image.height * scale_factor)
        
         
        
        resized_image = new_image.resize((new_width, new_height), Image.LANCZOS)
        
        
        # Calculate where to paste the image on the canvas
        paste_x = ((canvas_width - new_width) // 2 ) + 15
        paste_y = (canvas_height - new_height) // 2
        final_image = Image.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 0))
        final_image.paste(resized_image, (paste_x, paste_y), mask=resized_image.split()[3] if "A" in resized_image.getbands() else None)
        
        # Customize the output filename as needed
        final_image_filename = os.path.splitext(filename)[0] + '.png'  # Convert to PNG
        
        # Save the final image
        final_image_path = os.path.join(output_folder, final_image_filename)
        final_image.save(final_image_path, "PNG")
                

            
start_time = time.time()
input_folder= r"C:\Users\Richard\Documents\We Cloud Data\Watches\AWS\a_lange_sohne_downloads\original_images"
output_folder=r"C:\Users\Richard\Documents\We Cloud Data\Watches\AWS\a_lange_sohne_downloads\resized_images_2"
resize_and_center_images(input_folder,output_folder)
end_time = time.time()     
elapsed_time = end_time - start_time
print(f"Time taken: {elapsed_time} seconds")                     
