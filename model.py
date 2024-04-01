import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights, vgg16, VGG16_Weights


class ResnetLocal(nn.Module):
    def __init__(self):
        super(ResnetLocal, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Sequential(nn.Linear(in_features=512, out_features=64),
                                       nn.ReLU(),
                                       nn.Linear(in_features=64, out_features=2))
    def forward(self, x):
        return self.resnet(x)
    

class EncoderAndClassifier:
    # encoder = nn.Sequential(*list(resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).children())[:-1])
    # encoder = nn.Sequential(*list(resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).children())[:-1])
    modified_vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    encoder = nn.Sequential(*list(modified_vgg.features.children()))
    # num_features = resnet18().fc.in_features
    # num_features = resnet50().fc.in_features
    num_features = vgg16().classifier[0].in_features

    modified_vgg.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(num_features, num_features//4),
        nn.BatchNorm1d(num_features//4),
        nn.ReLU(),
        # nn.Linear(num_features//2, num_features//4),
        # nn.BatchNorm1d(num_features//4),
        # nn.ReLU(),
        nn.Linear(num_features//4, 400)
    )

    simple_classification = modified_vgg.classifier

    # simple_classification = nn.Sequential(
    #     nn.Flatten(),
    #     nn.Linear(num_features, num_features//2),
    #     nn.BatchNorm1d(num_features//2),
    #     nn.ReLU(),
    #     nn.Linear(num_features//2, num_features//4),
    #     nn.BatchNorm1d(num_features//4),
    #     nn.ReLU(),
    #     nn.Linear(num_features//4, 400)
    #     # nn.Linear(num_features, 400)
    # )
    
class CustomNetwork(nn.Module):
    def __init__(self, encoder, classification):
        super(CustomNetwork, self).__init__()

        self.encoder = encoder
        if self.encoder is None:
            self.encoder = EncoderAndClassifier.encoder

        for param in self.encoder.parameters():
            param.requires_grad = False

        self.mse_loss = nn.MSELoss()

        self.classification = classification
        if self.classification is None:
            self.classification = EncoderAndClassifier.simple_classification
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
        x = x.reshape(x.size(0), -1)  # Flatten the tensor while keeping the batch dimension
        output = self.classify(x)
        return output
