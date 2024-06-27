from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os
from PIL import Image
import cv2
import numpy as np




def crop_to_content_with_alpha(image):
    """
    Crop an image based on its transparency (alpha channel) to retain only the main content.
    
    Parameters:
    - image (PIL.Image): The input image to be processed.

    Returns:
    - PIL.Image: The cropped image, or the original image if no transparent pixels are detected.
    """
    
    # Ensure the image is in RGBA mode to handle transparency.
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
        
    # Split the channels and obtain the alpha (transparency) channel.
    alpha_channel = image.split()[3]
    
    # Check for any transparent pixels.
    if alpha_channel.getextrema()[0] < 255:
        # Get the bounding box of the content based on the transparency.
        bbox = alpha_channel.getbbox()
        
        # If a bounding box exists, crop the image.
        if bbox:
            cropped_image = image.crop(bbox)
            return cropped_image

    # If there are no transparent regions or if cropping isn't possible, return the original image.
    return image


def crop_to_content_without_alpha(image_path):
    """
    Crops an image to its main content using contours. This function is particularly
    suitable for images without an alpha channel.
    
    Args:
        image_path (str): Path to the input image.

    Returns:
        PIL.Image.Image: Cropped image.
    """
    # Load the image using PIL
    image = Image.open(image_path)
    
    # Convert the PIL image to OpenCV format (BGRA)
    open_cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGRA)
    
    # Convert the BGRA image to grayscale
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    
    # Apply a Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Dilate the edges to enhance them
    dilated = cv2.dilate(edges, None, iterations=2)
    
    # Find the contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Assuming the largest contour corresponds to the main object of interest
    # You can add logic here to select the desired contour if there are multiple
    x, y, w, h = cv2.boundingRect(contours[0])
    
    # Crop the image using the bounding rectangle of the contour
    cropped_image = open_cv_image[y:y+h, x:x+w]
    
    # Convert the cropped image from BGR to RGB
    cropped_image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
    
    # Convert the cropped image from OpenCV format to PIL format
    pil_cropped_image = Image.fromarray(cropped_image_rgb)
    
    return pil_cropped_image


def resize_image(cropped_image, canvas_width, canvas_height, fixed_margin):
    """
    Resize a given image, maintaining its aspect ratio, and center it on a predefined canvas.
    
    Parameters:
    - cropped_image (PIL.Image): The cropped image to be resized.
    - canvas_width (int): Width of the output canvas.
    - canvas_height (int): Height of the output canvas.
    - fixed_margin (int): The margin space from the canvas borders.

    Returns:
    - PIL.Image: The resized and centered image on the canvas.
    """
    
    # Calculate the aspect ratio of the cropped image.
    aspect_ratio = cropped_image.width / cropped_image.height
    
    # Define the maximum dimensions for the image after considering the margins.
    max_width = canvas_width - 2 * fixed_margin
    max_height = canvas_height - 2 * fixed_margin
    
    # Determine the scale factor to ensure the image fits within the canvas.
    scale_factor = min(max_width / cropped_image.width, max_height / cropped_image.height)
    
    # Calculate the new dimensions for the image.
    new_width = int(cropped_image.width * scale_factor)
    new_height = int(cropped_image.height * scale_factor)
    
    # Resize the cropped image based on the new dimensions.
    resized_image = cropped_image.resize((new_width, new_height), Image.LANCZOS)
    
    # Calculate the coordinates to center the resized image on the canvas.
    paste_x = ((canvas_width - new_width) // 2) + 15
    paste_y = (canvas_height - new_height) // 2
    
    # Create a transparent canvas.
    final_image = Image.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 0))
    
    # Paste the resized image onto the canvas.
    final_image.paste(resized_image, (paste_x, paste_y), mask=resized_image.split()[3] if "A" in resized_image.getbands() else None)

    return final_image

# for single thread process if resources are limited
# def process_images_in_folder(input_folder, output_folder, canvas_width=1000, canvas_height=1000, fixed_margin=10):
#     """
#     Process all the images in a given folder.
    
#     This function processes each image in the input folder by cropping and resizing it. 
#     The processed image is then saved in the output folder with the same name but in PNG format.
    
#     Parameters:
#     - input_folder (str): Path to the input directory containing images.
#     - output_folder (str): Path to the directory where processed images will be saved.
#     - canvas_width (int): Width of the output canvas (default: 1000).
#     - canvas_height (int): Height of the output canvas (default: 1000).
#     - fixed_margin (int): Margin from the canvas borders (default: 10).
#     """
    
#     # Create the output directory if it doesn't exist.
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png', '.jpeg', '.avif','webp'))]

#     for filename in tqdm(image_files, desc="Processing images"):
#         image_path = os.path.join(input_folder, filename)
#         image = Image.open(image_path).convert('RGBA')
        
        
#         # Check if the image has any transparency.
#         if image.mode == 'RGBA' and image.split()[3].getextrema()[0] < 255:
#             cropped_image = crop_to_content_with_alpha(image)
#         else:
#             cropped_image = crop_to_content_without_alpha(image_path)
                
#         final_image = resize_image(cropped_image, canvas_width, canvas_height, fixed_margin)
#         final_image_filename = os.path.splitext(filename)[0] + '.png'
#         final_image_path = os.path.join(output_folder, final_image_filename)
#         final_image.save(final_image_path, "PNG")

#Multicore process
def process_image(image_path, output_folder, canvas_width, canvas_height, fixed_margin):
    """
    Process a single image: crop, resize and save it.
    """
    filename = os.path.basename(image_path)
    image = Image.open(image_path).convert('RGBA')
    
    # Check if the image has any transparency.
    if image.mode == 'RGBA' and image.split()[3].getextrema()[0] < 255:
        cropped_image = crop_to_content_with_alpha(image)
    else:
        cropped_image = crop_to_content_without_alpha(image_path)
            
    final_image = resize_image(cropped_image, canvas_width, canvas_height, fixed_margin)
    final_image_filename = os.path.splitext(filename)[0] + '.png'
    final_image_path = os.path.join(output_folder, final_image_filename)
    final_image.save(final_image_path, "PNG")

def process_images_in_folder_concurrent(input_folder, output_folder, canvas_width=1000, canvas_height=1000, fixed_margin=10):
    """
    Process all images in a given folder concurrently using ThreadPoolExecutor.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png', '.jpeg', '.avif', 'webp'))]
    image_paths = [os.path.join(input_folder, filename) for filename in image_files]

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_image, image_path, output_folder, canvas_width, canvas_height, fixed_margin) for image_path in image_paths]
        
        # Initialize the progress bar with the total number of tasks
        with tqdm(total=len(futures), desc="Processing images") as progress_bar:
            for future in as_completed(futures):
                try:
                    future.result()  # Getting the result to catch exceptions if any
                except Exception as e:
                    print(f"An error occurred: {e}")
                finally:
                    # Update the progress bar by one for each completed future
                    progress_bar.update(1)



if __name__ == "__main__":
    input_folder = r"C:\Users\Richard\Downloads\Rolex\images_not_on_S3"
    output_folder = r"C:\Users\Richard\Downloads\Rolex\resized_images"
    process_images_in_folder_concurrent(input_folder, output_folder)

  