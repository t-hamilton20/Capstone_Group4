import json
import os

from PIL import Image

dsDir = 'data/Complete/Images/'
train_folder = ''
annotations_folder = 'data/Complete/mtsd_v2_fully_annotated/annotations'
output_dir = 'data/Complete/augmented/'

# Create a subfolder for extracted images if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

annotation_files = [os.path.join(annotations_folder, file) for file in
                    os.listdir(os.path.join(annotations_folder)) if file.endswith('.json')]

index = 0
labels = []

# Save sign annotations to a .txt file in the annotations subfolder
annotation_file_path = os.path.join(output_dir, 'sign_annotation.txt')
with open(annotation_file_path, 'w') as f:
    for annotation in annotation_files:
        index += 1
        fileName = os.path.splitext(annotation)[0]
        img_filename = os.path.basename(fileName) + '.jpg'
        imgDir = os.path.join(dsDir, train_folder, img_filename)

        # Check if the image file exists
        if not os.path.exists(imgDir):
            print(f"Image file not found: {imgDir}")
            continue

        with open(annotation, 'r') as file:
            data = json.load(file)

        full_image = Image.open(imgDir)

        # Extract sign information
        signs = data['objects']

        print(f"Image {index}/{len(annotation_files)}")

        i = 0
        for sign in signs:
            bbox = sign['bbox']
            label = sign['label']
            if label != 'other-sign':

                if label not in labels:
                    labels.append(label)

                xminTemp, yminTemp, ymaxTemp, xmaxTemp = bbox['xmin'], bbox['ymin'], bbox['ymax'], bbox['xmax']
                xmax = max(xmaxTemp, xminTemp)
                ymax = max(ymaxTemp, yminTemp)
                xmin = min(xminTemp, xmaxTemp)
                ymin = min(yminTemp, ymaxTemp)
                step_x = 4
                step_y = 4
                
                for row in range(step_x):
                    for col in range(step_y):

                        # Extract the base filename without the directory
                        base_filename = os.path.basename(fileName)

                        # Print or process the extracted information as needed
                        img_file = os.path.basename(fileName) + ".jpg"
                        label_index = labels.index(label)
                        entry = f"{base_filename}_{i}_{row}_{col}.jpg  Class: {label_index}  Coordinates: ({xmin+row*2},{ymin+col*2},{ymax+col*2},{xmax+row*2})"
                        print(entry, file=f)

                        # Crop the region from the full image
                        extracted_region = full_image.crop((xmin+row*2, ymin+col*2, xmax+row*2, ymax+col*2))

                        # Save the extracted region to the subfolder
                        extracted_image_path = os.path.join(output_dir, f"{base_filename}_{i}.jpg")
                        extracted_region.save(extracted_image_path)
                        i += 1

labels_file_path = os.path.join(output_dir, 'labels.txt')
with open(labels_file_path, 'w') as f:
    for i, label in enumerate(labels):
        f.write(f"{i}: {label}\n")
    
    f.close()