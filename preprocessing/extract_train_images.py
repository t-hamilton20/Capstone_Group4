from data_augmentation import extract_images

raw_images_dir = 'data/Complete/train/'
annotations_dir = 'data/Complete/annotations'
output_dir = 'data/Extracted/train/'
class_names_file = 'data/class_names.txt'
sliding_window_step = 5 # number of pixels the sliding window moves each step
min_image_size = 0 # minimum height and width of image
step_x = 3 # number of times the sliding window moves in the x direction
step_y = 3 # number of times the sliding window moves in the y direction
# Increase in number of images will be step_x*step_y, ex: for 2 and 2, a 4x increase in images

extract_images(raw_images_dir, annotations_dir, output_dir, class_names_file, sliding_window_step, min_image_size, step_x, step_y, '', 1)