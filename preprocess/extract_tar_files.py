import os
import shutil
import tarfile

# Define the paths
source_path = "/mnt/alxd-v1-r2-code/datasets/A2D2/compressed"
destination_path = "/mnt/alxd-v1-r2-code/datasets/A2D2/extracted"
local_temp_path = "./tmp_a2d2_extraction"

# Ensure the local temporary directory exists
if not os.path.exists(local_temp_path):
    os.makedirs(local_temp_path)

# Function to copy files
def copy_file(src, dst):
    shutil.copy2(src, dst)
    print(f"Copied {src} to {dst}")

# Function to extract tar files
def extract_tar_files(source, destination, temp):
    tar_files = [file for file in os.listdir(source) if file.endswith(".tar")]
    total_files = len(tar_files)

    print(f"Found {total_files} tar files to extract.")
    
    for idx, file_name in enumerate(tar_files, start=1):
        print(f"Processing file {idx} of {total_files}: {file_name}")
        
        # Construct full file path
        file_path = os.path.join(source, file_name)
        
        # Copy tar file to local temporary directory
        local_file_path = os.path.join(temp, file_name)
        print(f"Copying {file_name} to local temporary directory.")
        copy_file(file_path, local_file_path)
        print(f"Copied {file_name} to local temporary directory.")
        
        # Define extraction path
        extract_path = os.path.join(temp, os.path.splitext(file_name)[0])
        
        # Ensure the local extraction directory exists
        if not os.path.exists(extract_path):
            os.makedirs(extract_path)
            print(f"Created local directory: {extract_path}")

        # Open and extract tar file locally
        try:
            with tarfile.open(local_file_path) as tar:
                tar.extractall(path=extract_path)
                print(f"Successfully extracted {file_name} to local directory.")
                
            # Move extracted files to destination path
            final_destination_path = os.path.join(destination, os.path.splitext(file_name)[0])
            if not os.path.exists(final_destination_path):
                os.makedirs(final_destination_path)
            print(f"Moving extracted files to {final_destination_path}.")
            shutil.move(extract_path, final_destination_path)
            print(f"Moved extracted files to {final_destination_path}")
            
            # Clean up local tar file
            os.remove(local_file_path)
            print(f"Removed local copy of {file_name}.")
            
        except Exception as e:
            print(f"Failed to extract {file_name} due to error: {e}")

    print("Extraction process completed.")

# Run the extraction function
if __name__ == "__main__":
    extract_tar_files(source_path, destination_path, local_temp_path)