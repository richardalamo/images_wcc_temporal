import os
import shutil

def get_matching_names(failed_folder_path, original_folder_path):
    # Get file names without extension from the first folder
    names_set1 = set(os.path.splitext(filename)[0] for filename in os.listdir(failed_folder_path))

    # Get file names without extension from the second folder
    names_set2 = set(os.path.splitext(filename)[0] for filename in os.listdir(original_folder_path))

    # Return the intersection of the two sets
    return names_set1.intersection(names_set2)


def copy_matching_files(original_folder_path, final_folder_path, matching_names):
    # Ensure the final folder exists
    if not os.path.exists(final_folder_path):
        os.makedirs(final_folder_path)

    # Iterate over all files in the original folder
    for file in os.listdir(original_folder_path):
        # Extract the name without extension
        name_without_extension = os.path.splitext(file)[0]
        
        # Check if this file is in the matching names set
        if name_without_extension in matching_names:
            original_file = os.path.join(original_folder_path, file)
            destination_file = os.path.join(final_folder_path, file)
            shutil.copyfile(original_file, destination_file)

# Define your folder paths
failed_folder_path = r"C:\Users\Richard\Documents\We Cloud Data\Watches\Students- Scrapes 09-03-23\Richa\Bulova\final_failed"
original_folder_path = r"C:\Users\Richard\Documents\We Cloud Data\Watches\Students- Scrapes 09-03-23\Richa\Bulova\original_images"
final_folder_path = r"C:\Users\Richard\Documents\We Cloud Data\Watches\Students- Scrapes 09-03-23\Richa\Bulova\final_failed_originals"

matching_names= get_matching_names(failed_folder_path,original_folder_path)
# Copy the matching files
copy_matching_files(original_folder_path, final_folder_path, matching_names)
