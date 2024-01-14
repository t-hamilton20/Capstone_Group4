import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from custom_dataset import SignDataset


def get_device(is_cuda: str):
    if (is_cuda.lower() == 'y' and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


def transform():
    transform = transforms.Compose([
        transforms.Resize(size=(150, 150)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform


def test_single_image(model, image, device, topk=5):
    model.eval()
    image = image.to(device)
    image = image.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image)

    # Apply softmax to get probabilities
    probabilities = F.softmax(output, dim=1)

    # Get the top k predictions
    top_probabilities, top_classes = torch.topk(probabilities, topk)

    return top_probabilities.squeeze().cpu().numpy(), top_classes.squeeze().cpu().numpy()


def main(args):
    device = get_device(args.cuda)
    print("Device: ", device)

    rootDir = "./data/Extracted_Images/"

    test_dataset = SignDataset(root_dir=rootDir, train=False, transform=transform())
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    numClasses = 400

    # Load pre-trained ResNet-18
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, numClasses)
    model.to(device)

    model_path = "./data/models/"+args.s
    # Load the trained model weights
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    count = 0
    Top5Count = 0
    Top1Count = 0

    # Test the model on a single image
    for idx, (images, labels) in enumerate(test_loader):
        # Assuming only one image is loaded in the batch
        image, label = images[0], labels[0]

        # Test the model on the single image
        probabilities, predicted_classes = test_single_image(model, image, device, topk=5)

        if label == predicted_classes[0]:
            Top1Count += 1
            Top5Count += 1
        elif label in predicted_classes:
            Top5Count += 1

        count += 1
        print(count)

    print(f"Top 1 Accuracy: {Top1Count}/{count}")
    print(f"Top 5 Accuracy: {Top5Count}/{count}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', type=str, help='Path to the trained model weights')
    parser.add_argument('-cuda', type=str, default='Y', help="Whether to use CPU or Cuda, use Y or N")
    args = parser.parse_args()
    main(args)
