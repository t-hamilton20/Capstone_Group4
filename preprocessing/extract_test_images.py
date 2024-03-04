from data_augmentation import extract_images

raw_images_dir = 'data/Complete/val/'
annotations_dir = 'data/Complete/annotations'
output_dir = 'data/Extracted/test/'
class_names_file = 'class_names.txt'
preexisting_class_names_file = 'data/class_names.txt'
sliding_window_step = 0
min_image_size = 0
step_x = 0 
step_y = 0 

extract_images(raw_images_dir, annotations_dir, output_dir, class_names_file, sliding_window_step, min_image_size, step_x, step_y, preexisting_class_names_file)