import argparse
import torch
from matplotlib import pyplot as plt

from model import CustomNetwork
from torchvision import transforms
from custom_dataset import SignDataset
from preprocessing.data_augmentation import read_class_names
from attack_module.attack import attack

def test_transform():
    transform_list = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0, 0, 0], std=[0.5, 0.5, 0.5])
    ]
    return transforms.Compose(transform_list)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Arguments to pass to the test module')
    parser.add_argument('-cuda', type=str, default='cuda:1', help='device')
    parser.add_argument('-s', type=str, default='./data/models/6_mapillary_vgg_all_classes_brightness.pth', help='weight path')
    parser.add_argument('-i', type=int, default=126, help='image index to test')

    argsUsed = parser.parse_args()
    return argsUsed

if __name__ == "__main__":
    args = parse_arguments()

    root_dir = "./data/Extracted/"

    # Load the dataset without using DataLoader
    test_dataset = SignDataset(root_dir=root_dir, train=False, transform=test_transform())

    # Select a single image by index
    img, label = test_dataset[args.i]  # args.i is the image index
    img = img.unsqueeze(0)  # Add batch dimension
    img = attack("cpu", img, True, False, False, False, False)

    model = CustomNetwork(None, None)
    model.load_state_dict(torch.load(args.s))
    model.eval()

    classes = read_class_names('./data/Extracted/test/class_names.txt')

    with torch.no_grad():
        outputs = model(img)
        _, predicted_top5 = torch.topk(outputs.data, 5, dim=1)
        top5_predictions = [classes[predicted] for predicted in predicted_top5[0]]

        image = transforms.functional.to_pil_image(img.squeeze(0).cpu())
        plt.imshow(image)
        plt.title(f"Ground Truth: Class {classes[label]}")
        plt.axis('off')
        plt.show()

    print(f"Top 5 Predictions: {top5_predictions}")
