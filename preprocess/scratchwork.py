from PIL import Image
import os

# Define the dimensions and the output path
width, height = 1920, 1208
output_directory = "/mnt/alxd-v1-r2-code/datasets/A2D2/images"
output_path = os.path.join(output_directory, "blank_image.png")

# Ensure the output directory exists
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Create a blank (white) image
image = Image.new('RGB', (width, height), color=(255, 255, 255))

# Save the image
image.save(output_path)
print(f"Blank image created and saved to {output_path}")
