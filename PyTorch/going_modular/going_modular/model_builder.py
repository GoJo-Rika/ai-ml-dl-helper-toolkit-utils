"""Contains PyTorch model code to instantiate a TinyVGG model."""

import torch
import torchvision
from torch import nn


class TinyVGG(nn.Module):
    """
    Creates the TinyVGG architecture.

    Replicates the TinyVGG architecture from the CNN explainer website in PyTorch.
    See the original architecture here: https://poloclub.github.io/cnn-explainer/

    Args:
        input_shape: An integer indicating number of input channels.
        hidden_units: An integer indicating number of hidden units between layers.
        output_shape: An integer indicating number of output units.

    """

    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Where did this in_features shape come from?
            # It's because each layer of our network compresses and changes the shape of our inputs data.
            nn.Linear(in_features=hidden_units * 13 * 13, out_features=output_shape),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x
        # return self.classifier(self.block_2(self.block_1(x))) # <- leverage the benefits of operator fusion


class EfficientNet(nn.Module):
    """
    A class to create an EfficientNet model with a custom classifier head.

    Args:
        model_name (str): The name of the EfficientNet model (e.g., "effnetb0").
        num_classes (int): The number of output classes for the classifier.

    """

    def __init__(self, model_name: str, num_classes: int) -> None:
        super().__init__()
        self.name = model_name

        # Select EfficientNet variant and corresponding weights, input features, and dropout rate
        if model_name == "effnetb0":
            weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
            model = torchvision.models.efficientnet_b0(weights=weights)
            in_features = 1280
            dropout_rate = 0.2
        elif model_name == "effnetb2":
            weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
            model = torchvision.models.efficientnet_b2(weights=weights)
            in_features = 1408
            dropout_rate = 0.3
        else:
            # Raise error if unsupported model name is provided
            msg = f"Model '{model_name}' is not supported. Choose 'effnetb0' or 'effnetb2'."
            raise ValueError(msg)

        # Freeze all parameters in the feature extractor to prevent them from training
        for param in model.features.parameters():
            param.requires_grad = False

        # Assign the feature extractor and average pooling layers from the pretrained model
        self.features = model.features
        self.avgpool = model.avgpool

        # Define a new classifier head for custom number of output classes
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_features=in_features, out_features=num_classes),
        )

        print(f"[INFO] Created new {self.name} model with {num_classes} output classes.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pass input through feature extractor
        x = self.features(x)
        # Apply average pooling
        x = self.avgpool(x)
        # Flatten the output for the classifier
        x = torch.flatten(x, 1)
        # Pass through the classifier head
        x = self.classifier(x)
        return x
