import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class SignDetector(nn.Module):
    def __init__(self):
        super(SignDetector, self).__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()

        with torch.no_grad():
            self.out = self.backbone(torch.zeros(1, 3, 256, 256))

        num_features = self.out.shape[1]
        print(f"Num features: {num_features}")

        # Modify only the fc layer
        self.fc_layers = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 400)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc_layers(x)
        return x
