import os
import cv2

def determine_brightness(image_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to load.")

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the average pixel intensity
    average_intensity = cv2.mean(gray_image)[0]
    print("AVERAGE INTESITY: ", average_intensity)

    if average_intensity > 95:
        return 1
    else:
        return 0

def adjust_brightness(image_path, brightness_factor):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to load.")

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

    if brightness_category == 0:
        print("DARK")
    # Dark -> Bright
        brightened_image = adjust_brightness(image_path, 2.0)
        cv2.imwrite(file_name_without_extension + "_bright" + file_extension, brightened_image)

    else:
        print("BRIGHT")
    # Bright -> Dark
        darkened_image = adjust_brightness(image_path, 0.5)
        cv2.imwrite(file_name_without_extension + "_dark" + file_extension, darkened_image)

image_path = 'data/Complete/augmented_9_no_small/_GcRTRDQiYyn0bQzdByW7A_0_0_1.jpg'
generate_all_brightnesses(image_path)
