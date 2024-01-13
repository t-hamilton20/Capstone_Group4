import argparse
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from custom_dataset import SignDataset
import matplotlib.pyplot as plt
import os


def check_folders():
    # Define the main folder name
    main_folder = "data"

    # Check if the main folder exists, if not, create it
    if not os.path.exists(main_folder):
        os.mkdir(main_folder)
        print(f"Created folder: {main_folder}")

    # Define subfolder names
    subfolders = ["models", "loss_plots", "Sign_Dataset"]

    # Loop through subfolder names, create if they don't exist
    for folder in subfolders:
        folder_path = os.path.join(main_folder, folder)
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
            print(f"Created folder: {folder_path}")


def save_loss_plot(losses_train: list, save_path: str):
    # Plot training losses
    plt.plot([i for i in range(len(losses_train))], losses_train, label='Training Loss')

    # Set the title and labels
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    # Show the legend
    # plt.legend()

    save_path = "data/loss_plots/" + save_path

    # Save the plot
    plt.savefig(save_path)
    plt.close()


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


def validate(model, val_loader, device, criterion):
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)  # Calculate the loss
            val_loss += loss.item()
            _, predicted = outputs.max(1)  # Get the class with the highest score
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100 * correct / total
    average_loss = val_loss / len(val_loader)

    return accuracy, average_loss


def train(n_epochs: int, optimizer: torch.optim.Optimizer, model: nn.Module, train_loader: DataLoader,
          device: torch.device, criterion: nn.Module, scheduler):
    print(f"Starting training at: {datetime.datetime.now()}")
    # Initialize model save dir
    model_dir = "data/models/" + args.s
    model.train()

    losses_train = []

    # Training loop
    for epoch in range(n_epochs):
        print(f"Commencing Epoch {epoch + 1}")
        loss_train = 0.0
        i = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            i += 1

            # Forward pass
            outputs = model(images)

            # Compute loss
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()

        # Print training loss after each epoch
        scheduler.step()
        # Calculate the average loss for this epoch
        avg_loss = loss_train / len(train_loader)
        losses_train.append(avg_loss)

        print(f"{datetime.datetime.now()}: Epoch [{epoch + 1}/{n_epochs}], Loss: {loss.item()}")

    # Save the trained model
    torch.save(model.state_dict(), model_dir)
    return losses_train


def main(args):
    check_folders()
    device = get_device(args.cuda)
    print("Device: ", device)

    rootDir = "./data/Extracted_Images/"

    train_dataset = SignDataset(root_dir=rootDir, train=True, transform=transform())
    # test_dataset = SignDataset(train=False, transform=transforms.ToTensor())

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.b, shuffle=True)
    # test_loader = DataLoader(dataset=test_dataset, batch_size=args.b, shuffle=False)

    numClasses = 400

    # Load pre-trained ResNet-18
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, numClasses)
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = ExponentialLR(optimizer, gamma=0.9)

    losses_train = train(args.e, optimizer, model, train_loader, device, criterion, scheduler)

    save_loss_plot(losses_train, args.p)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', type=int, help='Number of epochs')
    parser.add_argument('-b', type=int, help='Batch Size')
    parser.add_argument('-s', type=str, help='Model out path')
    parser.add_argument('-p', type=str, help='Loss Plot Path')
    parser.add_argument('-cuda', type=str, default='Y', help="Whether to use CPU or Cuda, use Y or N")
    args = parser.parse_args()
    main(args)
