import os
import pandas as pd
from rembg import remove
from PIL import Image
from io import BytesIO
import current_image_resize_tool

def list_files(folder_failed_images):
    return os.listdir(folder_failed_images)

def filter_files(files, folder_path):
    return [file for file in files if os.path.isfile(os.path.join(folder_path, file))]

def remove_extensions(files):
    return [os.path.splitext(file)[0] for file in files]

def find_matching_files(df, files_to_match):
    matching_files = []
    for name in df['File Names']:
        matching_file = next((file for file in files_to_match if name in file), None)
        if matching_file:
            matching_files.append(matching_file)
    return matching_files

def remove_background_for_images(image_list):
    """
    Remove the background of a list of images using rembg.

    Parameters:
    - image_list (list of PIL.Image): A list of input images to process.

    Returns:
    - list of PIL.Image: A list of processed images.
    """
    processed_images = []
    
    for input_image in image_list:
        input_image = input_image.convert("RGBA")
        bytes_img = BytesIO()
        input_image.save(bytes_img, format="PNG")
        
        output_bytes = remove(bytes_img.getvalue())
        output_image = Image.open(BytesIO(output_bytes))
        
        processed_images.append(output_image)
    
    return processed_images

def save_processed_images(matching_files, processed_images, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for matching_file, processed_image in zip(matching_files, processed_images):
        output_file_path = os.path.join(output_folder, matching_file)
        processed_image.save(output_file_path, "PNG")




if __name__ == "__main__":
    #Folder with the failed images
    folder_failed_images = r"C:\Users\Richard\Documents\We Cloud Data\Watches\Finishing_scrapes\Tudor\failing_1"
    #Folder of the original images( from the website)
    folder_to_match = r"C:\Users\Richard\Documents\We Cloud Data\Watches\Finishing_scrapes\Tudor\original_images"
    #Output folder
    output_folder = r"C:\Users\Richard\Documents\We Cloud Data\Watches\Finishing_scrapes\Tudor\temp"
    #resizing folder
    final_folder = r"C:\Users\Richard\Documents\We Cloud Data\Watches\Finishing_scrapes\Tudor\resized_2"

    file_names = list_files(folder_failed_images)
    file_names = filter_files(file_names, folder_failed_images)
    file_names_without_extension = remove_extensions(file_names)
    
    df = pd.DataFrame({'File Names': file_names_without_extension})
    
    file_names_to_match = list_files(folder_to_match)
    file_names_to_match = filter_files(file_names_to_match, folder_to_match)

    matching_files = find_matching_files(df, file_names_to_match)
    
    images_to_process = [Image.open(os.path.join(folder_to_match, file_name)) for file_name in matching_files]
    processed_images = remove_background_for_images(images_to_process)
    
    save_processed_images(matching_files, processed_images, output_folder)

    current_image_resize_tool.process_images_in_folder_concurrent(output_folder,final_folder)
   





