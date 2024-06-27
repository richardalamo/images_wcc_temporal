from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from PIL import Image
import io
import numpy as np
import logging
import pandas as pd
import time
import os
from selenium import webdriver



logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("download_image.log"),
        logging.StreamHandler()
    ]
)

# List to store information about missing images
missing_images = []

def setup_driver(chrome_options=None):
    # Set up Chrome options
    if chrome_options is None:
        chrome_options = Options()
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--window-size=1920,1080")

    # Set up driver
    webdriver_service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=webdriver_service, options=chrome_options)

def take_screenshot(driver, url):
    
    try:
        driver.get(url)  # Navigate to the URL
        screenshot_as_png = driver.get_screenshot_as_png()
        image = Image.open(io.BytesIO(screenshot_as_png))  # Take the screenshot
        logging.info(f"Screenshot taken for URL: {url}")
    except Exception as e:  # Make sure to capture the exception with `as e`
        logging.error(f"Failed to take screenshot: {e}")
        return None

    return image

def crop_image(image, threshold=(50, 50, 50, 255)):
    """
    Crops the given PIL Image object to remove any border around it where the RGB values are below the threshold.
    The alpha value is also considered in the threshold, allowing for the removal of pixels that are not fully opaque.
    """
    # Convert the image to a numpy array
    image_np = np.array(image)

    # Find all pixels where the RGB values are less than the threshold, and the alpha is less than 255
    border_pixels = np.all(image_np < threshold, axis=-1)

    # Find the bounding box of the non-border areas
    coords = np.argwhere(~border_pixels)  # Invert the condition to get non-border pixels
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1  # Add 1 because slice indexing is exclusive at the top

    # Crop the image to that bounding box ensuring the order is (left, upper, right, lower)
    cropped_image = image.crop((x0, y0, x1, y1))

    return cropped_image
    
def process_screenshots(df, output_path):
    driver = setup_driver()

    try:
        for i, row in df.iterrows():
            try:
                image = take_screenshot(driver, row['image_url'])
                if image is None:
                    raise Exception("Screenshot returned None")

                file_extension = os.path.splitext(row['image_url'])[1] if os.path.splitext(row['image_url'])[1] else '.png'
                image_save_path = os.path.join(output_path, f"{row['brand']}_{row['specific_model']}_{row['reference_number']}{file_extension}")
                image.save(image_save_path)  # Save the Image object, not the bytes
                logging.info(f"Downloaded {row['brand']}_{row['specific_model']}_{row['reference_number']}")

            except Exception as e:
                missing_images.append(f"{row['brand']}_{row['specific_model']}_{row['reference_number']}")
                logging.error(f"Failed to process image: {e}")

    finally:
        driver.quit()


# url= 'https://www.iwc.com/content/dam/rcq/iwc/bo/sh/9z/tu/uU/Ck/ks/cT/4M/GG/AA/bosh9ztuuUCkkscT4MGGAA.png.transform.global_image_png_320_2x.png'

# image= take_screenshot(url)
# cropped_image= crop_image(image)
# cropped_image.save("cropped_image.png")

if __name__ == "__main__":
    read_path = r"C:\Users\Richard\Downloads\IWC_Chrono24_Watchbase.csv"
    output_path = r"C:\Users\Richard\Downloads\IWC_cropped"
    missing_images = []  # Initialize the list to track missing images

    # Ensure the output directory exists, create it if not
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Read df file
    df = pd.read_csv(read_path)
    df = df[df['image_url'].apply(lambda x: isinstance(x, str))]
    df['specific_model']= df['specific_model'].apply(lambda x: x.replace("Add to my wishlist",''))
    df['specific_model'] = df['specific_model'].apply(lambda x: x.replace("\n", "").strip())
    # Process the entire dataframe of valid image URLs
    if not df.empty:
        process_screenshots(df, output_path)  # Pass the entire dataframe here

    # Write the possible errors
    with open("missing_images.txt", "w") as f:
        for item in missing_images:
            f.write(f"{item}\n")

    
    
       


