import torch
import torch.nn as nn
import torchvision.models as models


class MultiModalModel(nn.Module):
    def __init__(self, text_feature_size, num_classes):
        super(MultiModalModel, self).__init__()
        vgg19 = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        vgg19_features = vgg19.features
        self.image_model = nn.Sequential(
            vgg19_features,
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.text_model = nn.Sequential(
            nn.Linear(text_feature_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 + 4096, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, images, text_features):
        img_features = self.image_model(images)
        text_features = self.text_model(text_features)
        combined_features = torch.cat((img_features, text_features), dim=1)
        outputs = self.classifier(combined_features)
        return outputs
