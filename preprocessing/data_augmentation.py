'''
Loops through each annotation file in the given directory
For each file, extracts each sign image
For each sign, takes sub-images, using a sliding window approach to augment the number of images extracted
Ignores images smaller than min_image_size x min_image_size pixels 
Uses brightness module to generate dark or bright images 
'''

import json
import os
from PIL import Image
from .brightness import generate_all_brightnesses

raw_images_dir = 'data/Complete/Images/'
annotations_dir = 'data/Complete/mtsd_v2_fully_annotated/annotations'
output_dir = 'data/Complete/augmented_8_no_small/'
class_names_file = 'class_names.txt'
sliding_window_step = 5 # number of pixels the sliding window moves each step
min_image_size = 50 # minimum height and width of image
step_x = 3 # number of times the sliding window moves in the x direction
step_y = 3 # number of times the sliding window moves in the y direction
# Increase in number of images will be step_x*step_y, ex: for 2 and 2, a 4x increase in images

def extract_images(raw_images_dir, annotations_dir, output_dir, output_class_names_file, sliding_window_step, min_image_size, step_x, step_y, preexisting_class_names_file, brightness):
    # to disable the sliding window functionality, set sliding_window_step, step_x, and step_y to 0
    # to disable the minimum image size check, set min_image_size to 0
    # to disable preexisting class names, set preexisting_class_names_file to ''

    # Create a subfolder for extracted images if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get all annotated JSON files 
    annotation_files = [os.path.join(annotations_dir, file) for file in
                    os.listdir(os.path.join(annotations_dir)) if file.endswith('.json')]

    class_names = []
    skipped_counter = 0
    dark_counter = 0
    bright_counter = 0

    if preexisting_class_names_file:
        class_names = read_class_names(preexisting_class_names_file)

    # Save sign annotations to a .txt file in the annotations subfolder
    annotation_file_path = os.path.join(output_dir, 'annotations.txt')
    with open(annotation_file_path, 'w') as f:
        for index, annotation in enumerate(annotation_files):
            file_name = os.path.splitext(annotation)[0]
            image_filename = os.path.basename(file_name) + '.jpg'
            image_file_dir = os.path.join(raw_images_dir, image_filename)

            # Check if the image file exists
            if not os.path.exists(image_file_dir):
                print(f"Image file not found: {image_file_dir}")
                continue
        
            # Open JSON file with image data
            with open(annotation, 'r') as file:
                image_data = json.load(file)

            full_image = Image.open(image_file_dir)

            # Extract sign information
            signs = image_data['objects']

            print(f"Image {index}/{len(annotation_files)}")


            # Loop through each sign in the JSON file
            for i, sign in enumerate(signs):
                bbox = sign['bbox']
                label = sign['label']

                if label == 'other-sign': # skip over other-sign signs
                    continue 

                if label not in class_names and not preexisting_class_names_file:
                    class_names.append(label)

                xminTemp, yminTemp, ymaxTemp, xmaxTemp = bbox['xmin'], bbox['ymin'], bbox['ymax'], bbox['xmax']
                xmax = max(xmaxTemp, xminTemp) - sliding_window_step
                ymax = max(ymaxTemp, yminTemp) - sliding_window_step
                xmin = min(xminTemp, xmaxTemp) - sliding_window_step
                ymin = min(yminTemp, ymaxTemp) - sliding_window_step

                # Check the width and height using ymin, ymax, xmin, and xmax
                if ((ymax - ymin) < min_image_size or (xmax - xmin) < min_image_size) and min_image_size > 0:
                    print(f"Skipped: {os.path.basename(file_name)}_{i}.jpg (Width or Height < {min_image_size})")
                    skipped_counter += 1
                    continue
            
                # Extract the base filename without the directory
                base_filename = os.path.basename(file_name)
                
                if step_x and step_y and sliding_window_step > 0:
                    for row in range(step_x):
                        for col in range(step_y):
                            image_file_name = f"{base_filename}_{i}_{row}_{col}.jpg"
                            label_index = class_names.index(label)
                            entry = f"{image_file_name}  Class: {label_index}  Coordinates: ({xmin},{ymin},{ymax},{xmax})"
                            print(entry, file=f)

                            # Crop the region from the full image
                            extracted_region = full_image.crop((xmin+row*sliding_window_step, ymin+col*sliding_window_step, xmax+row*sliding_window_step, ymax+col*sliding_window_step))

                            # Save the extracted region to the subfolder
                            extracted_image_path = os.path.join(output_dir, image_file_name)
                            extracted_region.save(extracted_image_path)

                            new_file_type, new_file_name = generate_all_brightnesses(extracted_image_path)

                            if new_file_type == 1: bright_counter += 1
                            else: dark_counter += 1

                            new_entry = f"{new_file_name}  Class: {label_index}  Coordinates: ({xmin},{ymin},{ymax},{xmax})"
                            print(new_entry, file=f)



                else:
                    label_index = class_names.index(label)
                    entry = f"{base_filename}_{i}.jpg  Class: {label_index}  Coordinates: ({xmin},{ymin},{ymax},{xmax})"
                    print(entry, file=f)

                    # Crop the region from the full image
                    extracted_region = full_image.crop((xmin, ymin, xmax, ymax))

                    # Save the extracted region to the subfolder
                    extracted_image_path = os.path.join(output_dir, f"{base_filename}_{i}.jpg")
                    extracted_region.save(extracted_image_path)
                    
                    new_file_type, new_file_name = generate_all_brightnesses(extracted_image_path)

                    if new_file_type == 1: bright_counter += 1
                    else: dark_counter += 1

                    new_entry = f"{new_file_name}  Class: {label_index}  Coordinates: ({xmin},{ymin},{ymax},{xmax})"
                    print(new_entry, file=f)



    classes_file_path = os.path.join(output_dir, output_class_names_file)
    with open(classes_file_path, 'w') as f:
        for i, label in enumerate(class_names):
            f.write(f"{i}: {label}\n")
        f.close()

    print(f"Bright counter: {bright_counter}")
    print(f"Dark counter: {dark_counter}")
    print(f"Number of skipped images: {skipped_counter}")


def read_class_names(class_names_file):
    # Read labels from the preexisting file
    class_names = []
    with open(class_names_file, 'r') as file:
        for line in file:
            label = line.strip().split(': ')[1]
            class_names.append(label)

    return class_names