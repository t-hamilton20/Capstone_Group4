from PIL import Image
import json
import os


def resize_annotation(annotation_data, original_width, original_height, new_width, new_height):
    scale_x = new_width / original_width
    scale_y = new_height / original_height

    for obj in annotation_data['objects']:
        obj['bbox']['xmin'] *= scale_x
        obj['bbox']['xmax'] *= scale_x
        obj['bbox']['ymin'] *= scale_y
        obj['bbox']['ymax'] *= scale_y

    return annotation_data

# Set desired width and height
new_width = 3264
new_height = 2448

image_directory = '../data/Complete/Images'
annotation_directory = '../data/Complete/mtsd_v2_fully_annotated/annotations'

# New directories to save resized images and annotations
resized_image_directory = '../data/Complete/resized'
resized_annotation_directory = '../data/Complete/mtsd_v2_fully_annotated/resized_annotations'

processed_files = []
skipped_counter = 0

# Iterate through image directory
for image_filename in os.listdir(image_directory):
    image_path = os.path.join(image_directory, image_filename)
    annotation_filename = os.path.splitext(image_filename)[0] + '.json'
    annotation_path = os.path.join(annotation_directory, annotation_filename)

    # Load and resize annotation data
    with open(annotation_path, 'r') as f:
        annotation_data = json.load(f)
    
    has_valid_box = any(obj['label'] != 'other-sign' for obj in annotation_data['objects'])

    if has_valid_box:
        # Load image
        # image = Image.open(image_path)

        # # Resize image
        # resized_image = image.resize((new_width, new_height))

        # # Get original image dimensions
        # original_size = image.size
        # original_width = original_size[0]
        # original_height = original_size[1]

        # # Resize annotation
        # resized_annotation = resize_annotation(annotation_data, original_width, original_height, new_width, new_height)

        # # Save resized image
        # resized_image_path = os.path.join(resized_image_directory, 'resized_' + image_filename)
        # resized_image.save(resized_image_path)

        # # Save updated annotation
        # resized_annotation_path = os.path.join(resized_annotation_directory, 'resized_' + annotation_filename)
        # with open(resized_annotation_path, 'w') as f:
        #     json.dump(resized_annotation, f)
        
        processed_files.append('resized_' + os.path.splitext(image_filename)[0]) 
    
    else:
        skipped_counter += 1

print(f"Skipped Counter: {skipped_counter}")

with open("../data/Complete/mtsd_v2_fully_annotated/splits/resized_train.txt", "w") as f:
    for filename in processed_files:
        f.write(filename + "\n")