import os
import argparse
import datetime
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from custom_dataset import SignDataset  # Ensure this is your dataset class
from model import CustomResNet  # Adjust this to your model class
from torch.cuda.amp import GradScaler, autocast

def parse_arguments():
    parser = argparse.ArgumentParser(description='Arguments for training the model')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--device', default='cuda', help='Device to use for training')
    parser.add_argument('--save_path', type=str, default='model.pth', help='Path to save the trained model')
    parser.add_argument('--plot_path', type=str, default='loss_plot.png', help='Path to save the loss plot')
    return parser.parse_args()

def train_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def test_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def train_one_epoch(model, train_loader, loss_fn, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):

        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with autocast():
            outputs = model(images)
            loss = loss_fn(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        if i % 100 == 0:  # Adjust this value based on your dataset size and preference
            print(f"Batch {i}/{len(train_loader)}")

    return running_loss / len(train_loader.dataset)

def validate(model, val_loader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            running_loss += loss.item() * images.size(0)
    return running_loss / len(val_loader.dataset)

def main():
    args = parse_arguments()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    root_dir = "./data/Extracted_Images"

    train_dataset = SignDataset(root_dir=root_dir, transform=train_transform())
    val_dataset = SignDataset(root_dir=root_dir, transform=test_transform())

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = CustomResNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    loss_fn = torch.nn.CrossEntropyLoss()
    scaler = GradScaler()  # For mixed precision training

    losses_train, losses_val = [], []
    for epoch in range(args.epochs):
        print(f"{datetime.datetime.now()} Epoch {epoch+1}/{args.epochs}")
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, scaler, device)
        val_loss = validate(model, val_loader, loss_fn, device)
        scheduler.step()

        losses_train.append(train_loss)
        losses_val.append(val_loss)

        print(f"{datetime.datetime.now()} - Epoch {epoch+1} - Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        # Plotting training and validation loss
        plt.figure(figsize=(10, 5))
        plt.plot(losses_train, label='Training Loss')
        plt.plot(losses_val, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plotpath = os.path.join('data/loss_plots',f'{args.plot_path}')
        plt.savefig(plotpath)
        plt.close()

        # Saving model checkpoints
        filepath = os.path.join('data/models/intermediates', f"{epoch}_{args.save_path}")
        torch.save(model.state_dict(), filepath)

    # Saving the final model
    torch.save(model.state_dict(), args.save_path)
    print(f"Training completed. Final model saved to {args.save_path}")

if __name__ == '__main__':
    main()
