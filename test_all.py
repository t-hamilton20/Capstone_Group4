import argparse
import torch
from model import CustomResNet
from torchvision import transforms
from custom_dataset import SignDataset
from torch.utils.data import DataLoader


def test_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def parse_arguments():
    parser = argparse.ArgumentParser(description='Arguments to pass to the train module')
    parser.add_argument('-cuda', type=str, default='cuda', help='device')
    parser.add_argument('-s', type=str, default='model.pth', help='weight path')
    parser.add_argument('-b', type=int, default=1, help='batch size')

    argsUsed = parser.parse_args()
    return argsUsed


def main():
    args = parse_arguments()
    device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")

    test_dataset = SignDataset(root_dir='./data/Extracted_Images', train=False, transform=test_transform())
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.b, shuffle=False)

    model = CustomResNet()  # Initialize your model
    model.load_state_dict(torch.load(args.s, map_location=device))
    model.to(device)
    model.eval()

    correct_top1 = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            _, top1_preds = torch.max(outputs, 1)
            correct_top1 += (top1_preds == labels).sum().item()

            _, top5_preds = torch.topk(outputs, 5, dim=1)
            correct_top5 += torch.sum(torch.any(top5_preds == labels.unsqueeze(1), dim=1)).item()

            if i % 100 == 0:
                print(f"Processed {i * args.b}/{len(test_dataset)} samples...")

    top1_accuracy = 100 * correct_top1 / len(test_dataset)
    top5_accuracy = 100 * correct_top5 / len(test_dataset)

    print(f'Top-1 Accuracy: {top1_accuracy:.2f}%')
    print(f'Top-5 Accuracy: {top5_accuracy:.2f}%')

if __name__ == "__main__":
    main()
