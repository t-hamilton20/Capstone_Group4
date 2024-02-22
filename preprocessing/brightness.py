import os
import cv2

def determine_brightness(image_path):
    # Return dark if 0, bright if 1

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image {image_path} not found or unable to load.")

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the average pixel intensity
    average_intensity = cv2.mean(gray_image)[0]
    # print("AVERAGE INTESITY: ", average_intensity)

    if average_intensity > 95:
        return 1
    else:
        return 0

def adjust_brightness(image_path, brightness_factor):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image {image_path} not found or unable to load.")

    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Adjust the brightness by scaling the V channel (Value/Brightness)
    hsv_image[:, :, 2] = hsv_image[:, :, 2] * brightness_factor

    # Convert the image back to BGR color space
    adjusted_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    return adjusted_image

def generate_all_brightnesses(image_path):
    brightness_category = determine_brightness(image_path)
    # file_name, file_extension = os.path.splitext(image_path)
    
    file_name = os.path.basename(image_path)
    file_name_without_extension, file_extension = os.path.splitext(file_name)
    directory_path = os.path.dirname(image_path)

    if brightness_category == 0:
    # # Dark -> Bright
        brightened_image = adjust_brightness(image_path, 1.5)
        new_file_name = file_name_without_extension + "_bright" + file_extension
        new_file_path = os.path.join(directory_path, new_file_name)
        cv2.imwrite(new_file_path, brightened_image)

    else:
    # Bright -> Dark
        darkened_image = adjust_brightness(image_path, 0.5)
        new_file_name = file_name_without_extension + "_dark" + file_extension
        new_file_path = os.path.join(directory_path, new_file_name)
        cv2.imwrite(new_file_path, darkened_image)
    
    return brightness_category, new_file_name

# image_path = 'data/Complete/augmented_9_no_small/_GcRTRDQiYyn0bQzdByW7A_0_0_1.jpg'
# generate_all_brightnesses(image_path)
