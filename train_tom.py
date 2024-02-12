import os
import argparse
import datetime
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from custom_dataset import SignDataset
from model import EncoderAndClassifier, CustomNetwork
import torchsummary


def parse_arguments():
    parser = argparse.ArgumentParser(description='Arguments to pass to the train module')
    parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('-e', type=int, default=30, help='number of epochs')
    parser.add_argument('-b', type=int, default=512, help='batch size')
    parser.add_argument('-cuda', type=int, default='1', help='device')
    parser.add_argument('-s', type=str, default='weights.pth', help='weights path')
    parser.add_argument('-p', type=str, default='loss_plot.png', help='Path to save the loss plot')

    return parser.parse_args()


def train_transform():
    transform_list = [
        # transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        # transforms.CenterCrop((224, 224)),
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

num_classes = 350

def train(n_epochs, optimizer, model, loss_fn, train_loader, val_loader, scheduler, device):
    print(f"Starting training at: {datetime.datetime.now()}")

    losses_train = []
    losses_val = []

    # class_weights = calculate_class_weights(train_loader, num_classes)
    # class_weights = class_weights.to(device)

    for epoch in range(1, n_epochs + 1):
        model.train()
        loss_train = 0.0
        
        for img, labels in train_loader:
            img = img.to(device=device)
            labels = labels.to(device=device)
            optimizer.zero_grad()
            outputs = model(img)
            loss = loss_fn(outputs, labels)
            # loss = torch.nn.functional.cross_entropy(outputs, labels, weight=class_weights)
            loss.backward()
            optimizer.step()
            loss_train += loss.item()

        scheduler.step(loss_train)

        model.eval()
        loss_val = 0.0
        with torch.no_grad():
            for img_val, labels_val in val_loader:
                img_val = img_val.to(device=device)
                labels_val = labels_val.to(device=device)

                outputs_val = model(img_val)
                loss_val += loss_fn(outputs_val, labels_val).item()

        losses_train += [loss_train / len(train_loader)]
        losses_val += [loss_val / len(val_loader)]

        training_loss = loss_train / len(train_loader)
        validation_loss = loss_val / len(val_loader)
        print(f"{datetime.datetime.now()} Epoch {epoch}. Training Loss {training_loss}, Validation Loss {validation_loss}")

        plt.plot(losses_train, label='Training Loss')
        plt.plot(losses_val, label='Validation Loss')
        plt.legend()
        plt.savefig("data/loss_plots/" + args.p)
        plt.clf()

        filepath = os.path.join('data/models/intermediates', f"{epoch}_{args.s}")
        torch.save(model.state_dict(), filepath)


args = parse_arguments()
torch.cuda.empty_cache()

device = 'cuda'
if args.cuda == 0:
    device = 'cpu'
print(f'Device: {device}')

root_dir = "../data/Complete/"

train_dataset = SignDataset(root_dir=root_dir, train=True, transform=train_transform())
test_dataset = SignDataset(root_dir=root_dir, train=False, transform=test_transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=args.b, shuffle=True)
val_loader = DataLoader(dataset=test_dataset, batch_size=args.b, shuffle=False)

model = CustomNetwork(None, None)
# model.load_state_dict(torch.load('data/models/intermediates/11_no_small.pth'))
model.train()
model.to(device)
# print(torchsummary.summary(model, batch_size=args.b, input_size=(3, 224, 224)))

lr = args.lr
opt = torch.optim.Adam(params=model.parameters(), lr=lr)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=opt, factor=0.1, patience=5, verbose=True)
# sched = torch.optim.lr_scheduler.StepLR(optimizer=opt, gamma=0.1, step_size=15)
# sched = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt, gamma=0.95)
loss_fn = torch.nn.CrossEntropyLoss()

train(n_epochs=args.e, optimizer=opt, model=model, scheduler=sched, loss_fn=loss_fn, device=device, train_loader=train_loader, val_loader=val_loader)
model_dir = "data/models/" + args.s
model_state_dict = model.state_dict()
torch.save(model_state_dict, model_dir)
print("model saved")



