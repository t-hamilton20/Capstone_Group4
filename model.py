import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class CustomResNet(nn.Module):
    def __init__(self, num_classes=400):
        super(CustomResNet, self).__init__()
        # Load a pre-trained ResNet50 model
        self.base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        # Remove the original fully connected layer
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()

        # Additional fully connected layers with batch normalization
        self.fc_layers = nn.Sequential(
            nn.Linear(num_features, 512),  # First additional FC layer
            nn.BatchNorm1d(512),  # Batch Normalization
            nn.ReLU(inplace=True),  # Activation function
            nn.Dropout(0.5),  # Dropout for regularization
            nn.Linear(512, num_classes)  # Final FC layer that outputs the class scores
        )

    def forward(self, x):
        x = self.base_model(x)  # Pass input through the base model
        x = self.fc_layers(x)  # Pass base model output through the new FC layers
        return x
