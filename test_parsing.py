import json
import os

from PIL import Image

dsDir = 'data/Sign_Dataset/'
annotation_files = [os.path.join(dsDir, file) for file in
                    os.listdir(os.path.join(dsDir)) if file.endswith('.json')]

for annotation in annotation_files:
    fileName = os.path.splitext(annotation)[0]
    annoDir = fileName+'.json'
    print(annoDir)
    with open(annoDir, 'r') as file:
        data = json.load(file)

    imgDir = fileName+'.jpg'
    full_image = Image.open(imgDir)

    # Extract sign information
    signs = data['objects']

    # Iterate through each sign and extract bbox coordinates and class
    for sign in signs:
        bbox = sign['bbox']
        label = sign['label']
        if label != 'other-sign':
            xmin, ymin, ymax, xmax = bbox['xmin'], bbox['ymin'], bbox['ymax'], bbox['xmax']

            # Print or process the extracted information as needed
            print(f"Sign Class: {label}")
            print(f"Bounding Box Coordinates: xmin={xmin}, ymin={ymin}, ymax={ymax}, xmax={xmax}\n")

            # Crop the region from the full image
            extracted_region = full_image.crop((xmin, ymin, xmax, ymax))

            # Show the extracted region
            # extracted_region.show()
