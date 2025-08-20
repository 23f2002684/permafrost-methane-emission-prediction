import os

# Set the path to the folder containing your masks
# Based on your folder structure, this should be correct if you run the script from the MLPROJCORE root directory
mask_folder = 'data/masks'

# Check if the directory exists
if not os.path.exists(mask_folder):
    print(f"Error: The directory '{mask_folder}' was not found. Please check the path.")
else:
    # Loop through all files in the mask folder
    for filename in os.listdir(mask_folder):
        # Check if the file is a mask that needs renaming
        if 'mask_' in filename and filename.lower().endswith('.png'):
            # Define the old and new file paths
            old_path = os.path.join(mask_folder, filename)
            
            # Create the new filename by removing ALL 'mask_' instances
            new_name = filename.replace('mask_', '') 
            new_path = os.path.join(mask_folder, new_name)
            
            # Rename the file
            os.rename(old_path, new_path)
            print(f"Renamed '{filename}' to '{new_name}'")

    print("\nAll relevant mask files have been renamed successfully!")

