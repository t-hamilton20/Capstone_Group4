import os
import argparse
import datetime
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from custom_detection_dataset import CustomDetectionDataset
from model import EncoderAndClassifier, CustomNetwork
from detection_model import get_model_with_pretrained_backbone, get_simple_model, CustomFastRCNN
from dynamic_batch import DynamicBatcher
# import torchsummary


def parse_arguments():
    parser = argparse.ArgumentParser(description='Arguments to pass to the train module')
    parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('-e', type=int, default=30, help='number of epochs')
    parser.add_argument('-b', type=int, default=512, help='batch size')
    parser.add_argument('-cuda', type=int, default='1', help='device')
    parser.add_argument('-s', type=str, default='weights.pth', help='weights path')
    parser.add_argument('-p', type=str, default='loss_plot.png', help='Path to save the loss plot')

    return parser.parse_args()

def collate(batch):
    return tuple(zip(*batch))

def train_transform():
    transform_list = [
        # transforms.ToPILImage(),
        # transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        # transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0, 0, 0], std=[0.5, 0.5, 0.5])
    ]
    return transforms.Compose(transform_list)


test_transform = transforms.Compose([
    # transforms.Resize((224, 224)),
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

num_classes = 400

def train(n_epochs, optimizer, model, loss_fn, train_loader, val_loader, scheduler, device):
    print(f"Starting training at: {datetime.datetime.now()}")

    losses_train = []
    losses_val = []

    for epoch in range(1, n_epochs + 1):
        
        model.train()
        loss_train = 0.0

        for images, targets in train_loader:

            targets_list = []
            for t in targets:
                targets_list.extend(t)

            targets = targets_list

            images = torch.stack(images, dim=0)
            images = images.to(device=device)
            targets = [{k: v.to(device=device) for k, v in t.items()} for t in targets]
            
            optimizer.zero_grad()
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())

            loss.backward()
            optimizer.step()
            loss_train += loss.item()
        
        scheduler.step(loss_train)

        model.eval()
        loss_val = 0.0
        with torch.no_grad():
            for images_val, targets_val in val_loader:

                targets_val_list = []
                for t in targets_val:
                    targets_val_list.extend(t)

                targets_val = targets_val_list

                images_val = torch.stack(images_val, dim=0)
                images_val = images_val.to(device=device)
                targets_val = [{k: v.to(device=device) for k, v in t.items()} for t in targets_val]
                
                loss_val_dict = model(images_val, targets_val)
                loss = sum(loss for loss in loss_val_dict.values())

                loss_val += loss.item()

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

        filepath = os.path.join('data/detection_models/intermediates', f"{epoch}_{args.s}")
        torch.save(model.state_dict(), filepath)


args = parse_arguments()
torch.cuda.empty_cache()

device = 'cuda'
if args.cuda == 0:
    device = 'cpu'
print(f'Device: {device}')

root_dir = "../data/Complete/"
train_annotations_dir = "mtsd_v2_fully_annotated/resized_annotations"
train_image_dir = 'resized'
test_annotations_dir = "mtsd_v2_fully_annotated/resized_annotations_val"
test_image_dir = 'resized_val'
class_names_file = '../data/Complete/augmented_9_x_with_brightness/class_names.txt'

train_dataset = CustomDetectionDataset(root_dir=root_dir, annotation_dir=train_annotations_dir, image_dir=train_image_dir, class_names_file=class_names_file, train=True, transform=train_transform())
test_dataset = CustomDetectionDataset(root_dir=root_dir, annotation_dir=test_annotations_dir, image_dir=test_image_dir, class_names_file=class_names_file, train=False, transform=test_transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=args.b, shuffle=True, collate_fn=collate)
val_loader = DataLoader(dataset=test_dataset, batch_size=args.b, shuffle=False, collate_fn=collate)

# backbone = CustomNetwork(None, None)
# backbone.load_state_dict(torch.load('data/models/intermediates/30_mapillary_resnet18_all_classes_brightness.pth'))

model = get_simple_model(400)
# model = CustomFastRCNN(backbone)
# model = get_model_with_pretrained_backbone(num_classes=num_classes, backbone=backbone)
model.train()
model.to(device)
# print(torchsummary.summary(model, batch_size=args.b, input_size=(3, 224, 224)))

model_parameters = [p for p in model.parameters() if p.requires_grad]
lr = args.lr
opt = torch.optim.Adam(params=model_parameters, lr=lr)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=opt, factor=0.1, patience=5, verbose=True)
# sched = torch.optim.lr_scheduler.StepLR(optimizer=opt, gamma=0.1, step_size=15)
# sched = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt, gamma=0.95)
loss_fn = torch.nn.CrossEntropyLoss()

train(n_epochs=args.e, optimizer=opt, model=model, scheduler=sched, loss_fn=loss_fn, device=device, train_loader=train_loader, val_loader=val_loader)
model_dir = "data/detection_models/" + args.s
model_state_dict = model.state_dict()
torch.save(model_state_dict, model_dir)
print("model saved")



