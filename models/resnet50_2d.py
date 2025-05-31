import torch
import torch.nn as nn
import torchvision.models as models


class ResNet50(nn.Module):
    def __init__(self, weights='IMAGENET1K_V1'):
        super(ResNet50, self).__init__()

        # Load pretrained ResNet50
        self.resnet = models.resnet50(weights=weights)

        # Replace final FC layer with a binary classification layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, 1)  # Output = 1 logit for binary classification

    def forward(self, x):
        return self.resnet(x)  # Raw logits


# Test the model
if __name__ == "__main__":
    # Dummy batch of 4 RGB images, resized to 224x224
    dummy_input = torch.randn(4, 3, 224, 224)

    # Initialize the model
    model = ResNet50()

    # Forward pass
    output = model(dummy_input)

    # Print shape and raw output
    print("Output shape:", output.shape)  # Expect: (4, 1)
    print("Raw logits:\n", output)

    # Convert logits to probabilities with sigmoid
    probs = torch.sigmoid(output)
    print("Probabilities:\n", probs)
