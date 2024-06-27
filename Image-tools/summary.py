#################################################
#  This python code do the statistics of successfully resized images, failed images, and successfully downloaded images
#
#  Created by Andrew Xu
#
#  Revised by Liwei (Vivian) Kuang on March 14, 2024
#      1, Check the images in Original_Images direcotry instead of checking the no_empty of image_url in the brand_bulk.csv 
#         and mark the successfully downloaded images in the summary.csv
#         Even the image_url is not empty, the files may not be downloaded or downloaded with blank images which are invalid 
#      2, Set an variable 'brand' to easily change among the brands
#      3, Correct typo error of the get_iamge_index to get_image_index
#      4, Employ the variable brand in the path names which can easily swith among brands
#
#################################################

import os
import pandas as pd
import numpy as np

brand = 'Mido'
appended_file_path = f"G:/My Drive/Desktop/WatchData/brands/{brand}/{brand}_bulk.csv"
original_path = r'D:\Download\Original_Images'
resized_path = r'D:\Download\Resized_Images'
final_failed_path = r'D:\Download\final_failed_originals'

def get_image_index(resized_folder_path):
    '''
    This function will go throught each file in the desired folder, and firstly extract 
    the base file name, and then split the first part to get the index from defined folder.
    Then store the result to a list
    '''
    index_list = []
    for i in os.listdir(resized_folder_path):
        index_list.append(int(os.path.splitext(i)[0].split('_')[0]))
    return index_list

# to save the file
def to_save(path):
    image_summary.to_csv(path)
    return None

# get the inital data index, and store to a list
df_read = pd.read_csv(appended_file_path)
index = df_read.index.tolist()

# get the index of successfully resized images
resized_images = get_image_index(resized_path)

# get the index of final failed images
final_failed = get_image_index(final_failed_path)

# get the index of successfully downloaded ones 
original_images = get_image_index(original_path)

# convert to dataframe
image_summary = pd.DataFrame(index)

# add a column that present successfully resized
image_summary['Resized_Image(OK)'] = image_summary[0].isin(resized_images).astype(int)

# add a column that present final_failed
image_summary['Failling'] = image_summary[0].isin(final_failed).astype(int)

# add a column to present missing image_url ones
image_summary['Downloaded_Images(OK)'] = image_summary[0].isin(original_images).astype(int)

# rename columns
image_summary.rename(columns={0:'Index'},inplace=True)

image_summary.set_index('Index',inplace=True)


if __name__ == "__main__":
    # define folder path for saving this summay file
    summary_path = f"G:/My Drive/Desktop/WatchData/brands/{brand}/{brand}_summary.csv"
    to_save(summary_path)
