from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os
from PIL import Image
import cv2
import numpy as np

# Threshold Variables
BLUR_THRESHOLD = 100  # Variance of Laplacian threshold. Typical values: min=10, avg=100, max=200
EDGE_COUNT_THRESHOLD = 2000  # Edge count threshold. Typical values: min=500, avg=1000, max=2000
WHITE_BG_THRESHOLD = 240  # RGB threshold to consider a pixel as white. Typical values: min=200, avg=240, max=255
WHITE_BG_PERCENTAGE = 0.9  # Percentage of white pixels in the corners to consider background as white. Typical values: min=0.5, avg=0.9, max=1.0

def crop_to_content_with_alpha(image):
    """ 
    Crop image to content based on transparency (alpha channel).
    
    Parameters:
    image (PIL.Image): The image to be cropped.
    
    Returns:
    PIL.Image: The cropped image if transparency is present, otherwise the original image.
    """
    # Ensure the image is in RGBA mode to access the alpha channel
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    # Extract the alpha channel (transparency) from the image
    alpha_channel = image.split()[3]
    
    # Check if there's any transparency in the image
    if alpha_channel.getextrema()[0] < 255:
        # Get the bounding box of the non-transparent content
        bbox = alpha_channel.getbbox()
        if bbox:
            # Crop the image to the bounding box
            cropped_image = image.crop(bbox)
            return cropped_image
    
    # Return the original image if no transparency is found
    return image

def crop_to_content_without_alpha(image_path):
    """ 
    Crop image to content based on contours (for images without alpha channel).
    
    Parameters:
    image_path (str): The path to the image to be cropped.
    
    Returns:
    PIL.Image: The cropped image.
    """
    # Open the image using PIL
    image = Image.open(image_path)
    
    # Convert the PIL image to an OpenCV image (NumPy array) with alpha channel
    open_cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGRA)
    
    # Convert the OpenCV image to grayscale
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Perform edge detection using the Canny algorithm
    edges = cv2.Canny(blurred, 50, 150)
    
    # Dilate the edges to close gaps
    dilated = cv2.dilate(edges, None, iterations=2)
    
    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get the bounding box of the largest contour
    x, y, w, h = cv2.boundingRect(contours[0])
    
    # Crop the OpenCV image to the bounding box
    cropped_image = open_cv_image[y:y+h, x:x+w]
    
    # Convert the cropped OpenCV image back to RGB mode for PIL compatibility
    cropped_image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
    
    # Convert the cropped image to a PIL image
    pil_cropped_image = Image.fromarray(cropped_image_rgb)
    
    # Return the cropped PIL image
    return pil_cropped_image

def is_image_vertical(image):
    """ Check if the image is vertical (portrait orientation) """
    width, height = image.size
    return height > width

def rotate_to_vertical(image):
    """ Rotate image to make it vertical if it's not already """
    if not is_image_vertical(image):
        image = image.rotate(90, expand=True)
    return image

def has_transparency(image):
    """ Check if the image has transparency (alpha channel) """
    if image.mode == 'RGBA':
        alpha_channel = image.split()[3]
        if alpha_channel.getextrema()[0] < 255:
            return True
    return False

def is_background_mostly_white(image, threshold=WHITE_BG_THRESHOLD, percentage=WHITE_BG_PERCENTAGE):
    """ Check if the background of the image is mostly white """
    image = image.convert("RGBA")
    width, height = image.size
    datas = image.getdata()

    corners = [
        (0, 0),  # Top-left
        (width - 1, 0),  # Top-right
        (0, height - 1),  # Bottom-left
        (width - 1, height - 1)  # Bottom-right
    ]

    white_pixels = 0
    total_pixels = len(corners)

    for corner in corners:
        pixel = datas[corner[1] * width + corner[0]]
        if pixel[0] > threshold and pixel[1] > threshold and pixel[2] > threshold:
            white_pixels += 1

    return (white_pixels / total_pixels) >= percentage

def is_blurry_variance_of_laplacian(image, threshold=BLUR_THRESHOLD):
    """ Check if the image is blurry using the variance of Laplacian method """
    open_cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGR)
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    variance_of_laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance_of_laplacian < threshold

def is_blurry_edge_count(image, threshold=EDGE_COUNT_THRESHOLD):
    """ Check if the image is blurry using edge count method """
    open_cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGR)
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_count = np.count_nonzero(edges)
    return edge_count < threshold

def resize_image(cropped_image, canvas_width, canvas_height, fixed_margin):
    """ 
    Resize the image to fit within the specified canvas dimensions,
    maintaining the aspect ratio and centering the image within the canvas.
    
    Parameters:
    cropped_image (PIL.Image): The image to be resized.
    canvas_width (int): The width of the canvas.
    canvas_height (int): The height of the canvas.
    fixed_margin (int): The margin to maintain around the image.
    
    Returns:
    PIL.Image: The final image with the resized image centered on the canvas.
    """
    
    # Calculate the aspect ratio of the original image
    aspect_ratio = cropped_image.width / cropped_image.height
    
    # Calculate the maximum width and height the image can have within the canvas, accounting for margins
    max_width = canvas_width - 2 * fixed_margin
    max_height = canvas_height - 2 * fixed_margin
    
    # Determine the scale factor to resize the image while maintaining its aspect ratio
    scale_factor = min(max_width / cropped_image.width, max_height / cropped_image.height)
    
    # Calculate the new dimensions of the resized image
    new_width = int(cropped_image.width * scale_factor)
    new_height = int(cropped_image.height * scale_factor)
    
    # Resize the image using the calculated dimensions
    resized_image = cropped_image.resize((new_width, new_height), Image.LANCZOS)
    
    # Calculate the position to paste the resized image on the canvas to center it
    paste_x = (canvas_width - new_width) // 2
    paste_y = (canvas_height - new_height) // 2
    
    # Create a new transparent canvas with the specified dimensions
    final_image = Image.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 0))
    
    # Paste the resized image onto the canvas, handling transparency if present
    final_image.paste(resized_image, (paste_x, paste_y), mask=resized_image.split()[3] if "A" in resized_image.getbands() else None)
    
    # Return the final image with the resized image centered on the canvas
    return final_image

def process_image(image_path, transparent_output_folder, white_bg_output_folder, canvas_width, canvas_height, fixed_margin):
    """ Process a single image: crop, resize, and save it """
    filename = os.path.basename(image_path)
    image = Image.open(image_path).convert('RGBA')

    # Check if the image has transparency
    if has_transparency(image):
        cropped_image = crop_to_content_with_alpha(image)
        output_folder = transparent_output_folder
        print(f"Processing {filename} with transparency")
    else:
        # Check for blurriness for images without alpha
        if is_blurry_variance_of_laplacian(image):
            print(f"Skipping {filename} due to blurriness (variance of Laplacian)")
            return
        
        if is_blurry_edge_count(image):
            print(f"Skipping {filename} due to blurriness (edge count)")
            return

        # Check if the background is mostly white
        if is_background_mostly_white(image):
            cropped_image = crop_to_content_without_alpha(image_path)
            output_folder = white_bg_output_folder
            print(f"Processing {filename} with white background")
        else:
            print(f"Skipping {filename} due to non-white background")
            return

    # Rotate and resize the image
    cropped_image = rotate_to_vertical(cropped_image)
    final_image = resize_image(cropped_image, canvas_width, canvas_height, fixed_margin)
    final_image_filename = os.path.splitext(filename)[0] + '.png'
    final_image_path = os.path.join(output_folder, final_image_filename)
    final_image.save(final_image_path, "PNG")

def process_images_in_folder_concurrent(input_folder, transparent_output_folder, white_bg_output_folder, canvas_width=1000, canvas_height=1000, fixed_margin=10):
    """ Process all images in a given folder concurrently """
    if not os.path.exists(transparent_output_folder):
        os.makedirs(transparent_output_folder)
    if not os.path.exists(white_bg_output_folder):
        os.makedirs(white_bg_output_folder)

    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png', '.jpeg', '.avif', 'webp'))]
    image_paths = [os.path.join(input_folder, filename) for filename in image_files]

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_image, image_path, transparent_output_folder, white_bg_output_folder, canvas_width, canvas_height, fixed_margin) for image_path in image_paths]
        with tqdm(total=len(futures), desc="Processing images") as progress_bar:
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"An error occurred: {e}")
                finally:
                    progress_bar.update(1)

if __name__ == "__main__":
    input_folder = r"C:\Users\Richard\Documents\We Cloud Data\Watches\Finishing_scrapes\IWC\images"
    transparent_output_folder = r"C:\Users\Richard\Documents\We Cloud Data\Watches\Finishing_scrapes\IWC\transparent_images"
    white_bg_output_folder = r"C:\Users\Richard\Documents\We Cloud Data\Watches\Finishing_scrapes\IWC\white_bg_images"
    process_images_in_folder_concurrent(input_folder, transparent_output_folder, white_bg_output_folder)
