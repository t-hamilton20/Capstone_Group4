import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ResnetLocal(nn.Module):
    def __init__(self, num_classes=400):
        super(ResnetLocal, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        in_channels = 3  # Assuming RGB images
        self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(in_features, num_classes)  # Update to num_classes
        )

    def forward(self, x):
        return self.resnet(x)

class EncoderAndClassifier:
    encoder = nn.Sequential(*list(resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).children())[:-1])
    num_features = resnet18().fc.in_features
    simple_classification = nn.Sequential(
        nn.Flatten(),
        nn.Linear(num_features, 400)
    )

class CustomNetwork(nn.Module):
    def __init__(self, encoder=None, classification=None):
        super(CustomNetwork, self).__init__()

        self.encoder = encoder if encoder is not None else EncoderAndClassifier.encoder

        for param in self.encoder.parameters():
            param.requires_grad = False

        self.mse_loss = nn.MSELoss()

        self.classification = classification if classification is not None else EncoderAndClassifier.simple_classification
        self.init_classification_weights(mean=0.0, std=0.1)

    def init_classification_weights(self, mean, std):
        for param in self.classification.parameters():
            nn.init.normal_(param, mean=mean, std=std)

    def encode(self, x):
        return self.encoder(x)

    def classify(self, x):
        return self.classification(x)

    def forward(self, x):
        x = self.encoder(x)
        output = self.classify(x)
        return output
