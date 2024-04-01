import random
import torch
import numpy as np
import cv2


def attack(device, batch_tensor, add_rects: bool, rotate_imgs: bool, fish_img: bool, dented: bool, noisy: bool):
    batch_np = batch_tensor.cpu().numpy()
    batch_np = np.transpose(batch_np, (0, 2, 3, 1))

    processed_batch = []
    for img_np in batch_np:  # Process each image in the batch
        if dented:
            img_np = apply_dent(img_np)
        if add_rects:
            img_np = add_rectangles(img_np, width=random.randint(5, 60), height=random.randint(5, 60),
                                    num_rectangles=1)
        if rotate_imgs:
            img_np = rotate_images(img_np)
        if fish_img:
            img_np = apply_fisheye(img_np)
        if noisy:
            img_np = apply_noise(img_np)

        processed_batch.append(img_np)

    # Convert the processed batch back to tensor
    processed_batch_np = np.array(processed_batch)
    processed_batch_np = np.transpose(processed_batch_np, (0, 3, 1, 2))
    processed_tensor = torch.from_numpy(processed_batch_np).to(device)  # Ensure tensor is on the correct device
    return processed_tensor

def single_attack(device, single_tensor, add_rects: bool, rotate_imgs: bool, fish_img: bool, dented: bool, noisy: bool):
    img_np = single_tensor.cpu().numpy()
    img_np = np.transpose(img_np, (1, 2, 0))  # Transpose to (height, width, channels)

    if dented:
        img_np = apply_dent(img_np)
    if add_rects:
        img_np = add_rectangles(img_np, width=random.randint(20, 100), height=random.randint(20, 100),
                                num_rectangles=1)
    if rotate_imgs:
        img_np = rotate_images(img_np)
    if fish_img:
        img_np = apply_fisheye(img_np)
    if noisy:
        img_np = apply_noise(img_np)

    img_np = np.transpose(img_np, (2, 0, 1))  # Transpose back to (channels, height, width)
    processed_tensor = torch.from_numpy(img_np).to(device)  # Ensure tensor is on the correct device
    return processed_tensor

def add_rectangles(in_img_np, width: int, height: int, num_rectangles: int = 1):
    #BLACK
    #color = (256, 256, 256)
    #WHITE
    color = (1, 1, 1)
    # in_img_np is now expected to be a numpy array
    image = cv2.resize(in_img_np, (224, 224))
    image_height, image_width, _ = image.shape

    for i in range(num_rectangles):
        x = random.randint(0, image_width - width)
        y = random.randint(0, image_height - height)
        cv2.rectangle(image, (x, y), (x + width, y + height), color, thickness=-1)

    return image


def rotate_images(in_img, max_degrees: int = 90, min_degrees: int = 1):
    image = in_img
    image = cv2.resize(image, (224, 224))
    image_height, image_width, _ = image.shape

    degrees = random.randint(min_degrees, max_degrees)
    rotation_matrix = cv2.getRotationMatrix2D((image_width / 2, image_height / 2), degrees, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (image_width, image_height))

    return rotated_image


def apply_fisheye(in_img_np, distortion_strength=0.5):
    image_np = cv2.resize(in_img_np, (224, 224))
    height, width, _ = image_np.shape
    # Create a map for the distortion
    map_x = np.zeros((height, width), dtype=np.float32)
    map_y = np.zeros((height, width), dtype=np.float32)

    # Calculate the distortion
    for y in range(height):
        for x in range(width):
            # Normalize coordinates to the range [-1, 1]
            normalized_x = (x - width / 2.0) / (width / 2.0)
            normalized_y = (y - height / 2.0) / (height / 2.0)

            r = np.sqrt(normalized_x ** 2 + normalized_y ** 2)
            theta = np.arctan2(normalized_y, normalized_x)

            # Apply the fisheye distortion
            distorted_r = r * (1 + distortion_strength * r ** 2)

            # Map back to pixel coordinates
            distorted_x = (distorted_r * np.cos(theta) + 1) * (width / 2.0)
            distorted_y = (distorted_r * np.sin(theta) + 1) * (height / 2.0)

            map_x[y, x] = distorted_x
            map_y[y, x] = distorted_y

    # Remap the original image using the distortion maps
    distorted_image = cv2.remap(image_np, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    return distorted_image


def apply_dent(image_np, dent_strength=0.8, dent_radius=50, dent_center=None):
    height, width, _ = image_np.shape

    # If no dent center is provided, choose a random point within the image
    if dent_center is None:
        dent_center = (np.random.randint(dent_radius, width - dent_radius),
                       np.random.randint(dent_radius, height - dent_radius))

    # Create a map for the distortion
    map_x = np.zeros((height, width), dtype=np.float32)
    map_y = np.zeros((height, width), dtype=np.float32)

    for y in range(height):
        for x in range(width):
            # Calculate the distance from the center of the dent
            distance = np.sqrt((x - dent_center[0]) ** 2 + (y - dent_center[1]) ** 2)
            if distance < dent_radius:
                # Apply the dent effect within the dent radius
                normalized_distance = distance / dent_radius
                radial_distortion = dent_strength * (1 - np.cos(normalized_distance * np.pi)) / 2

                displacement = radial_distortion * (1 - normalized_distance)

                # Displace the current pixel towards the dent center
                map_x[y, x] = (1 - displacement) * x + displacement * dent_center[0]
                map_y[y, x] = (1 - displacement) * y + displacement * dent_center[1]
            else:
                # Outside the dent radius, pixels remain at their original position
                map_x[y, x] = x
                map_y[y, x] = y

    # Remap the original image using the distortion maps
    distorted_image = cv2.remap(image_np, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    return distorted_image


def apply_noise(image_np):
    std_dev = 0.075
    gauss_noise = np.random.randn(*image_np.shape) * std_dev
    gauss_noise = gauss_noise.astype(image_np.dtype)
    gn_img = cv2.add(image_np, gauss_noise)

    return gn_img
