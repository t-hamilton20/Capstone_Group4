import cv2
import os
import random

def add_rectangles(images_dir, width: int, height: int, num_rectangles: int = 1, color: tuple = (0, 0, 0)):
    print(os.listdir(images_dir))
    for file in os.listdir(images_dir):
        print(file)
        image = os.path.join(images_dir, file)
        image = cv2.imread(image)
        image = cv2.resize(image, (512, 512))
        image_height, image_width, _ = image.shape
        
        for i in range(num_rectangles):
            x = random.randint(0, image_width - width)
            y = random.randint(0, image_height - height)

            cv2.rectangle(image, (x, y), (x + width, y + height), color, thickness=-1)

        output = f"outputs/{file}_attacked.jpg"
        cv2.imwrite(output, image)
        print("Saved: ", output)


def rotate_images(images_dir, max_degrees: int = 355, min_degrees: int = 1):
    for file in os.listdir(images_dir):
        image = os.path.join(images_dir, file)
        image = cv2.imread(image)
        image = cv2.resize(image, (512, 512))
        image_height, image_width, _ = image.shape

        degrees = random.randint(min_degrees, max_degrees)
        rotation_matrix = cv2.getRotationMatrix2D((image_width / 2, image_height / 2), degrees, 1)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (image_width, image_height))

        output = f"outputs/{file}_attacked.jpg"
        cv2.imwrite(output, rotated_image)
        print("Saved: ", output)

rotate_images(images_dir='images/')
add_rectangles('outputs/', 50, 75, num_rectangles=2)
