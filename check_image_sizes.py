import os
from PIL import Image

# Define the path to the folder containing the images
folder_path = "../data/Complete/Images"

# Create a dictionary to keep track of image sizes and their counts
size_counts = {}

# Loop through each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Open the image file
        with Image.open(os.path.join(folder_path, filename)) as img:
            # Get the size of the image
            img_size = img.size

            # Increment the count for this size or initialize it to 1 if it's the first occurrence
            size_counts[img_size] = size_counts.get(img_size, 0) + 1

# Print the size counts
for size, count in size_counts.items():
    print(f"Size: {size}, Count: {count}")
