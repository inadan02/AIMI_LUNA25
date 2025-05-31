import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0


class EfficientNetB0(nn.Module):
    def __init__(self, weights='IMAGENET1K_V1'):
        super(EfficientNetB0, self).__init__()

        # Load pretrained EfficientNetB0
        self.model = efficientnet_b0(weights=weights)

        # Replace classifier with binary output
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 1)  # Output single logit for binary classification
        )

    def forward(self, x):
        return self.model(x)
if __name__ == "__main__":
    # Simulate a batch of 4 RGB images of size 224x224
    dummy_input = torch.randn(4, 3, 224, 224)

    # Instantiate model
    model = EfficientNetB0()

    # Forward pass
    output = model(dummy_input)
    print("Output shape:", output.shape)  # (4, 1)
    print("Raw logits:\n", output)

    # Convert logits to probabilities
    probs = torch.sigmoid(output)
    print("Probabilities:\n", probs)
