import os
import argparse
import datetime
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from custom_dataset import SignDataset
from model import CustomNetwork  # Make sure this is your correct model import

def parse_arguments():
    parser = argparse.ArgumentParser(description='Arguments to pass to the train module')
    parser.add_argument('-lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('-e', type=int, default=40, help='number of epochs')
    parser.add_argument('-b', type=int, default=512, help='batch size')
    parser.add_argument('-cuda', type=int, default=1, help='device')
    parser.add_argument('-s', type=str, default='Res50Test.pth', help='weights path')
    parser.add_argument('-p', type=str, default='loss_plot.png', help='Path to save the loss plot')
    parser.add_argument('-resume', type=str, default='', help='path to resume checkpoint (default: none)')
    return parser.parse_args()

def train_transform():
    transform_list = [
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0, 0, 0], std=[0.5, 0.5, 0.5])
    ]
    return transforms.Compose(transform_list)

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def calculate_class_weights(train_loader, num_classes):
    class_counts = torch.zeros(num_classes)
    total_samples = 0
    for _, labels in train_loader:
        class_counts += torch.bincount(labels, minlength=num_classes)
        total_samples += len(labels)
    class_weights = total_samples / (num_classes * class_counts)
    return class_weights

def train(n_epochs, optimizer, model, loss_fn, train_loader, val_loader, scheduler, device, start_epoch, losses_train, losses_val):
    for epoch in range(start_epoch, n_epochs + 1):
        print(f"{datetime.datetime.now()} - Epoch {epoch}/{n_epochs}")
        model.train()
        loss_train_epoch = 0.0

        for i, (img, labels) in enumerate(train_loader):
            if i % 100 == 0:
                print(f"Batch {i}/{len(train_loader)}")
            img, labels = img.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(img)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_train_epoch += loss.item()

        scheduler.step(loss_train_epoch)

        model.eval()
        loss_val_epoch = 0.0
        with torch.no_grad():
            for img_val, labels_val in val_loader:
                img_val, labels_val = img_val.to(device), labels_val.to(device)
                outputs_val = model(img_val)
                loss_val_epoch += loss_fn(outputs_val, labels_val).item()

        losses_train.append(loss_train_epoch / len(train_loader))
        losses_val.append(loss_val_epoch / len(val_loader))

        print(f"{datetime.datetime.now()} Epoch {epoch}. Training Loss {losses_train[-1]}, Validation Loss {losses_val[-1]}")

        plt.plot(losses_train, label='Training Loss')
        plt.plot(losses_val, label='Validation Loss')
        plt.legend()
        plt.savefig("data/loss_plots/" + args.p)
        plt.clf()

        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'loss_train': losses_train,
            'loss_val': losses_val
        }
        filepath = os.path.join('data/models/intermediates', f"{epoch}_{args.s}")
        torch.save(state, filepath)

def load_checkpoint(filepath, model, optimizer, scheduler):
    if os.path.isfile(filepath):
        print(f"Loading checkpoint '{filepath}'")
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        return checkpoint['epoch'] + 1, checkpoint['loss_train'], checkpoint['loss_val']
    else:
        print(f"No checkpoint found at '{filepath}', starting from scratch")
        return 1, [], []  # Start from the beginning if no checkpoint is found

args = parse_arguments()
torch.cuda.empty_cache()

device = 'cuda' if torch.cuda.is_available() and args.cuda > 0 else 'cpu'
print(f'Device: {device}')

root_dir = "./data/Extracted_Images/"
train_dataset = SignDataset(root_dir=root_dir, train=True, transform=train_transform())
val_dataset = SignDataset(root_dir=root_dir, train=False, transform=test_transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=args.b, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=args.b, shuffle=False)

model = CustomNetwork(None, None)  # Ensure you have the correct model initialization
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = torch.nn.DataParallel(model)
model.to(device)

lr = args.lr
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5, verbose=True)
loss_fn = torch.nn.CrossEntropyLoss()

if args.resume:
    start_epoch, losses_train, losses_val = load_checkpoint(args.resume, model, optimizer, scheduler)
else:
    start_epoch, losses_train, losses_val = 1, [], []

train(args.e, optimizer, model, loss_fn, train_loader, val_loader, scheduler, device, start_epoch, losses_train, losses_val)

# Saving the final model state
final_model_path = os.path.join("data/models/", args.s)
torch.save(model.state_dict(), final_model_path)
print(f"Final model saved to {final_model_path}")
