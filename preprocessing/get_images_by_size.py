import os
from PIL import Image

# Function to check image size
def check_image_size(image_path, target_size):
    with Image.open(image_path) as img:
        return img.size == target_size

# Function to read image filenames from a text file
def read_image_filenames(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]

# Function to write image filenames to a text file
def write_image_filenames(image_filenames, output_filename):
    with open(output_filename, 'w') as f:
        for filename in image_filenames:
            f.write(filename + '\n')

# Path to the text file containing the list of image filenames
input_filename = '../data/Complete/mtsd_v2_fully_annotated/splits/val.txt'

# Path to the directory containing the images
image_directory = '../data/Complete/Images'

# Size of the images to filter for
target_size = (4000, 3000)

# Read the list of image filenames
image_filenames = read_image_filenames(input_filename)

# Filter image filenames based on size
filtered_image_filenames = []

for filename in image_filenames:
    image_path = os.path.join(image_directory, filename) + '.jpg'
    
    if os.path.exists(image_path) and check_image_size(image_path, target_size):
        filtered_image_filenames.append(filename)
        print("APPENDING")

# Write filtered image filenames to a new text file
output_filename = '../data/Complete/mtsd_v2_fully_annotated/splits/filtered_val.txt'
write_image_filenames(filtered_image_filenames, output_filename)

print(f"Filtered images saved to {output_filename}")
