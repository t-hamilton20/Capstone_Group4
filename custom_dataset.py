import os
from PIL import Image
from torch.utils.data import Dataset


class SignDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.train = train

        # Define the folder based on the 'train' parameter
        if self.train:
            self.folder = 'train'
        else:
            self.folder = 'test'

        # Get the list of image files and corresponding labels
        self.image_files, self.labels = self.load_data()

    def load_data(self):
        image_files = []
        labels = []

        label_file = os.path.join(self.root_dir, self.folder, f'labels.txt')

        with open(label_file, 'r') as file:
            for line in file:
                # Split the line into image file, label, and class
                image_file, label, class_name = line.strip().split()
                image_files.append(os.path.join(self.root_dir, self.folder, image_file))

                # Update the label based on the class_name
                label = 1 if class_name == 'Car' else 0
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

        return img, label