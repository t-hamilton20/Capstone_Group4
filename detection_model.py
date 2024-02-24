import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import transforms
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights
    

# class DetectorAndClassifier:
#     classifier = nn.Sequential(*list(resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).children())[:-1])
#     # encoder = nn.Sequential(*list(resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).children())[:-1])
#     num_features = resnet18().fc.in_features
#     simple_classification = nn.Sequential(
#         nn.Flatten(),
#         nn.Linear(num_features, num_features//2),
#         nn.BatchNorm1d(num_features//2),
#         nn.ReLU(),
#         nn.Linear(num_features//2, num_features//4),
#         nn.BatchNorm1d(num_features//4),
#         nn.ReLU(),
#         nn.Linear(num_features//4, 400)
#         # nn.Linear(num_features, 400)
#     )
    
# class CustomDetectionNetwork(nn.Module):
#     def __init__(self, encoder, classification):
#         super(CustomDetectionNetwork, self).__init__()

#         self.encoder = encoder
#         if self.encoder is None:
#             self.encoder = DetectorAndClassifier.encoder

#         for param in self.encoder.parameters():
#             param.requires_grad = False

#         self.mse_loss = nn.MSELoss()

#         self.classification = classification
#         if self.classification is None:
#             self.classification = DetectorAndClassifier.simple_classification
#             self.init_classification_weights(mean=0.0, std=0.1)

#     def init_classification_weights(self, mean, std):
#         for param in self.classification.parameters():
#             nn.init.normal_(param, mean=mean, std=std)

#     def encode(self, x):
#         return self.encoder(x)

#     def classify(self, x):
#         return self.classification(x)

#     def forward(self, x):
#         x = self.encoder(x)
#         output = self.classify(x)
#         return output
    
# class FasterRCNNModel(torch.nn.Module):
#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)


# def get_model(num_classes):
#     # Load a pre-trained Faster R-CNN model
#     model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
#     # Replace the classifier with a new one that has num_classes classes
#     in_features = model.roi_heads.box_predictor.cls_score.in_features
#     model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    
#     return model

# # Example usage:
# # Define the number of classes in your dataset (including background)
# num_classes = 2  # Assuming you have 2 classes: background and object of interest

# # Instantiate the model
# model = get_model(num_classes)

# # Save the model
# torch.save(model.state_dict(), 'object_detection_model.pth')
def get_simple_model(num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained=True)

    # Modify the number of classes
    num_classes = num_classes  # Change this to your desired number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    return model

def get_model_with_pretrained_backbone(num_classes, backbone):
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)

    model = FasterRCNN(backbone=backbone, num_classes=num_classes, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler)
    return model

class CustomFastRCNN(nn.Module):
    def __init__(self, encoder_and_classifier):
        super(CustomFastRCNN, self).__init__()
        self.encoder = encoder_and_classifier.encoder
        self.classification = encoder_and_classifier.classification

        # Define the Region Proposal Network (RPN)
        num_feature_maps = 5  # This is an example value, adjust according to your backbone architecture

        # Define anchor sizes and aspect ratios for each feature map level
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )


        # Define Faster R-CNN model
        self.model = FasterRCNN(
            backbone=self.encoder,
            num_classes=400,
            rpn_anchor_generator=anchor_generator
        )

    def forward(self, images, targets=None):
        features = self.encoder(images)  # Get features from the backbone
        print(features.size())
        # Adjust the shape of the features if necessary
        # For example, if features.shape = [batch_size, num_channels, 7, 7], and subsequent layers expect a different shape:
        # features = adjust_shape(features)
        if self.training and targets is not None:
            loss_dict = self.model(images, targets)
            return loss_dict
        else:
            result = self.model(images)
            return result
