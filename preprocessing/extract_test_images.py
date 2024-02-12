from data_augmentation import extract_images

raw_images_dir = 'data/Complete/val/images/'
annotations_dir = 'data/Complete/mtsd_v2_fully_annotated/annotations'
output_dir = 'data/Complete/val/no_small/'
class_names_file = 'class_names.txt'
sliding_window_step = 0
min_image_size = 50
step_x = 0 
step_y = 0 

extract_images(raw_images_dir, annotations_dir, output_dir, class_names_file, sliding_window_step, min_image_size, step_x, step_y)