import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import mc3_18, MC3_18_Weights

class ResNet3D_MC3(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(ResNet3D_MC3, self).__init__()
        self.backbone = mc3_18(pretrained=pretrained)
        # self.backbone = mc3_18(weights=MC3_18_Weights.DEFAULT)

        # Replace the final fully connected layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        # self.backbone.fc = nn.Sequential(
        #     nn.Linear(in_features, 256),
        #     nn.ReLU(),
        #     nn.Dropout(0.4),
        #     nn.Linear(256, 64),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(64, num_classes)
        # )


    def forward(self, x):
        # x: shape [B, C, D, H, W]
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1, 1)  # Repeat grayscale channel to RGB
        return self.backbone(x).squeeze(-1)  # Output shape: [B]


if __name__ == "__main__":
    model = ResNet3D_MC3(num_classes=1, pretrained=True).cuda()
    dummy_input = torch.rand(2, 1, 64, 64, 64).cuda()
    output = model(dummy_input)
    print("Output shape:", output.shape)
