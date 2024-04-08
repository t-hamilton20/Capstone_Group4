import argparse
import torch
from model import EncoderAndClassifier, CustomNetwork
from torchvision import transforms
from custom_dataset import SignDataset
from torch.utils.data import DataLoader
from preprocessing.data_augmentation import read_class_names
from attack_module.attack import attack


def test_transform():
    transform_list = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0, 0, 0], std=[0.5, 0.5, 0.5])
    ]
    return transforms.Compose(transform_list)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Arguments to pass to the train module')
    parser.add_argument('-cuda', type=str, default='cuda', help='device')
    parser.add_argument('-s', type=str, default='./data/models/VGG_Attacked_Final.pth', help='weight path')
    parser.add_argument('-b', type=int, default=1, help='batch size')
    parser.add_argument('-checkpoint', type=bool, default=False, help='Checkpoint?')

    argsUsed = parser.parse_args()
    return argsUsed


if __name__ == "__main__":

    args = parse_arguments()

    device = torch.device(args.cuda if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    root_dir = "./data/Extracted/"
    test_dataset = SignDataset(root_dir=root_dir, train=False, transform=test_transform())
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.b, shuffle=False)

    model = CustomNetwork(None, None)

    checkpoint = torch.load(args.s, map_location=device)
    if args.checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    correct_top1 = 0
    correct_top5 = 0
    total = 0

    classes = read_class_names('./data/Extracted/test/class_names.txt')
    i = 0

    with torch.no_grad():
        for img, labels in test_loader:
            img = img.to(device)
            labels = labels.to(device)
            img = attack(device, img, False, False, False, False, False)
            outputs = model(img)
            _, predicted = torch.max(outputs.data, 1)
            # print(f"Predicted: {classes[predicted[0]]}, Label: {classes[labels[0]]}")
            total += labels.size(0)
            correct_top1 += (predicted == labels).sum().item()

            _, predicted_top5 = torch.topk(outputs.data, 5, dim=1)
            correct_top5 += sum([label in predicted_top5[i] for i, label in enumerate(labels)])
            if i % 100 == 0:
                print(f"{i}/{len(test_loader)}")
            i += 1

    top1_accuracy = 100 * correct_top1 / total
    top5_accuracy = 100 * correct_top5 / total

    print(f'Top-1 Accuracy: {top1_accuracy}%')
    print(f'Top-5 Accuracy: {top5_accuracy}%')
