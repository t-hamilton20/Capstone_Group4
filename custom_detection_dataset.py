import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
from preprocessing.data_augmentation import read_class_names


class CustomDetectionDataset(Dataset):
    def __init__(self, root_dir, annotation_dir, image_dir, class_names_file, transform=None, train=True):
        self.root_dir = root_dir
        self.annotation_dir = annotation_dir
        self.image_dir = image_dir
        self.class_names_file = class_names_file
        self.transform = transform
        self.train = train
        self.max_boxes = 15

        if self.train:
            self.image_list_file = '../data/Complete/mtsd_v2_fully_annotated/splits/resized_train.txt'

        else:
            self.image_list_file = '../data/Complete/mtsd_v2_fully_annotated/splits/resized_val.txt'
        
        with open(self.image_list_file, 'r') as f:
            self.image_list = [line.strip() for line in f]
        
        self.class_names = read_class_names(self.class_names_file)

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        annotation_file = os.path.join(self.root_dir, self.annotation_dir, self.image_list[idx]) + '.json'
        image_file = os.path.join(self.root_dir, self.image_dir, self.image_list[idx]) + '.jpg'
        
        with open(annotation_file, 'r') as f:
            annotation = json.load(f)
        
        image = Image.open(image_file).convert('RGB')            
        
        # Extract bounding boxes and labels
        boxes = []
        labels = []
        for obj in annotation['objects']:
            label = obj['label']

            if label == 'other-sign':
                continue

            numbered_label = self.class_names.index(label)
            labels.append(numbered_label)
            
            bbox = obj['bbox']
            xmin = bbox['xmin']
            ymin = bbox['ymin']
            xmax = bbox['xmax']
            ymax = bbox['ymax']
            boxes.append([xmin, ymin, xmax, ymax])
        
        # Pad bounding boxes if the number is less than the maximum allowed
        # num_boxes = len(boxes)
        # if num_boxes < self.max_boxes:
        #     pad_boxes = [[0, 0, 0, 0]] * (self.max_boxes - num_boxes)
        #     boxes.extend(pad_boxes)
        #     labels.extend([-1] * (self.max_boxes - num_boxes)) 

        # Convert everything into a PyTorch tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = [{'boxes' : boxes, 'labels' : labels}]
        
        if self.transform:
            image = self.transform(image)
                
        return image, target