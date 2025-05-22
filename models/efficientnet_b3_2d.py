import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3


class EfficientNetB3_MLP(nn.Module):
    def __init__(self, weights='IMAGENET1K_V1'):
        super(EfficientNetB3_MLP, self).__init__()

        # Load EfficientNet-B3 backbone with pretrained weights
        self.backbone = efficientnet_b3(weights=weights)

        # Extract number of input features from the original classifier
        in_features = self.backbone.classifier[1].in_features

        # Replace classifier with a custom MLP head
        # self.backbone.classifier = nn.Sequential(
        #     nn.Dropout(0.3),
        #     nn.Linear(in_features, 512),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(512),
        #     nn.Dropout(0.3),
        #     nn.Linear(512, 1)  # Single logit output for binary classification
        # )

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 1)
        )


    def forward(self, x):
        return self.backbone(x)  # Raw logits (no sigmoid here)

if __name__ == "__main__":
    model = EfficientNetB3_MLP()
    dummy_input = torch.randn(4, 3, 300, 300)  # EfficientNetB3 expects 300x300
    output = model(dummy_input)
    print("Output shape:", output.shape)  # Should be (4, 1)
    print("Logits:\n", output)
