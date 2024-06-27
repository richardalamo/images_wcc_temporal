import os

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

