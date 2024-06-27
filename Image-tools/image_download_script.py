import requests
from PIL import Image
from io import BytesIO
import pandas as pd
import logging
import os 
import mimetypes

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("download_image.log"),
        logging.StreamHandler()
    ]
)

def rename_files(directory):
    # Iterate over all the files in the directory
    for filename in os.listdir(directory):
        # Check if the file name contains a space
        if ' ' in filename:
            # Replace spaces with underscores
            new_filename = filename.replace(' ', '_')
            # Rename the file
            os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))
            print(f"Renamed '{filename}' to '{new_filename}'")


# List to store information about missing images
missing_images = []


def sanitize_filename(filename):
    """
    Sanitize the filename by replacing or removing invalid characters.
    """
    invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
    for char in invalid_chars:
        filename = filename.replace(char, '_')  # Replace with underscore or any other valid character
    return filename
   
   
def download_image(url, brand, model, sku, index, output_path):
    """
    Download an image based on the given URL, brand, model, and SKU.
    
    Parameters:
    ----------
    url : str
        URL of the image to download
    brand : str
        Brand of the watch
    model : str
        Model of the watch
    sku : str
        SKU of the watch
    index : int
        Unique index to avoid filename collisions
    output_path : str
        Directory to save the downloaded images

    Returns:
    -------
    None
    """

    logging.info(f"Attempting to download {index}_{brand}_{model}_{sku}")

    # Headers to mimic a real user request
    headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Safari/537.36"
    }
    
    # Check if the URL is a string
    if not isinstance(url, str):
        missing_images.append(f"Skipping {index}_{brand}_{model}_{sku} due to invalid URL.")
        return
    
    try:
        response = requests.get(url, headers=headers)
    except Exception as e:
        logging.error(f"Failed to send GET request: {e}")
        return
    
    # Check if the request was successful
    if response.status_code == 200:  # HTTP OK
        try:
            # Read and save the image
            image_data = BytesIO(response.content)
            image = Image.open(image_data)

            # Extract the file extension from the URL
            file_extension = mimetypes.guess_extension(response.headers.get('content-type'))

            if file_extension:
                sanitized_filename = sanitize_filename(f"{hash_url}{file_extension}")
                image_save_path = os.path.join(output_path, sanitized_filename)
                image.save(image_save_path)
                logging.info(f"Downloaded {index}_{brand}_{model}_{sku}")

            else:            
                logging.warning(f"Cannot determine file extension for {index}_{brand}_{model}_{sku}. Saving as PNG.")
                sanitized_filename = sanitize_filename(f"{hash_url}.png")
                image_save_path = os.path.join(output_path, sanitized_filename)
                image.save(image_save_path)             
           
        except Exception as e:
            logging.error(f"Failed to save image: {e}")
            missing_images.append(f"Failed to download {index}_{brand}_{model}_{sku}")
    else:
        missing_images.append(f"Failed to download {index}_{brand}_{model}_{sku}")

if __name__ == "__main__":
    read_path = r"/Users/richard/Downloads/Bulk_A_lange_2.csv"
    output_path = r"/Users/richard/Downloads/test_images"
    # Ensure the output directory exists, create it if not
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # read df file
    df = pd.read_csv(read_path)
    df = df[df['image_url'].apply(lambda x: isinstance(x, str))]

    #iterating on the dataframe
    if not df.empty:
        for i, row in df.iterrows():
            download_image(row['image_url'], row['brand'], row['specific_model'], row['reference_number'], i, output_path)

    # write the possible errors
    with open("missing_images.txt", "w") as f:
        for item in missing_images:
            f.write(f"{item}\n")

    rename_files(output_path)
    
