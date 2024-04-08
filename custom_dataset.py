import os
from PIL import Image
from torch.utils.data import Dataset
from attack_module.attack import attack
import random


class SignDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True, attack_train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.attack_train = attack_train

        # Define the folder based on the 'train' parameter
        if self.train:
            self.folder = 'train'
        else:
            self.folder = 'test'
        
        print(f"Using dataset located at {self.folder}")

        # Get the list of image files and corresponding labels
        self.image_files, self.labels = self.load_data()

    def load_data(self):
        image_files = []
        labels = []

        label_file = os.path.join(self.root_dir, self.folder, f'annotations.txt')

        with open(label_file, 'r') as file:
            for line in file:
                # Split the line into image file, label, and class
                image_file, _, label, _, coordinates = line.strip().split()
                image_files.append(os.path.join(self.root_dir, self.folder, image_file))

                labels.append(int(label))

        return image_files, labels

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        label = self.labels[idx]

        # Load image
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        if self.attack_train:
            rectBool = random.random() < 0.1
            rotnBool = random.random() < 0.1
            fishBool = random.random() < 0.1
            dentBool = random.random() < 0.1
            noiseBool = random.random() < 0.1
            img = attack('cuda', img.unsqueeze(0), rectBool, rotnBool, fishBool, dentBool, noiseBool).squeeze(0)

        return img, label
