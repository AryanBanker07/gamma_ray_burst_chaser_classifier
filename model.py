"""
Neural network architectures for Burst Chaser 3-class classification.
Backbones: ResNet-18 (default), ConvNeXt-Tiny, ResNet-34.
"""

from typing import Optional
import torch
import torch.nn as nn
from torchvision import models


class BurstChaserClassifier(nn.Module):
    """
    Vision classifier fine-tuned for classifying light-curve features into
    pulse, noise, or unclear.
    """

    def __init__(
        self,
        backbone_name: str = "resnet18",
        num_classes: int = 3,
        pretrained: bool = True,
        dropout: float = 0.3,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.backbone_name = backbone_name.lower()
        self.num_classes = num_classes

        if self.backbone_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            backbone = models.resnet18(weights=weights)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Identity()
            self.feature_extractor = backbone
        elif self.backbone_name == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            backbone = models.resnet34(weights=weights)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Identity()
            self.feature_extractor = backbone
        elif "convnext" in self.backbone_name:
            weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            backbone = models.convnext_tiny(weights=weights)
            in_features = backbone.classifier[2].in_features
            backbone.classifier[2] = nn.Identity()
            self.feature_extractor = backbone
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}. Choose from resnet18, resnet34, convnext_tiny.")

        # Custom MLP classification head with dropout and batch normalization
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def freeze_backbone(self, freeze: bool = True) -> None:
        """
        Freeze or unfreeze backbone feature extractor layers.
        """
        for param in self.feature_extractor.parameters():
            param.requires_grad = not freeze

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Input: (B, 3, 224, 224)
        Output: (B, num_classes) raw logits
        """
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits


class DualStreamBurstChaserClassifier(nn.Module):
    """
    Dual-stream (global + local) vision architecture.
    Stream 1 (Global): Full light curve capturing baseline noise floor, background count rate, and axes.
    Stream 2 (Local): Cropped region around the red elliptical marker capturing high-resolution pulse curvature.
    Features are concatenated into a 1024-dimensional embedding and classified via a joint MLP head.
    """

    def __init__(
        self,
        num_classes: int = 3,
        pretrained: bool = True,
        dropout: float = 0.3,
        hidden_dim: int = 256,
    ):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None

        # Global stream backbone
        global_backbone = models.resnet18(weights=weights)
        in_features_g = global_backbone.fc.in_features
        global_backbone.fc = nn.Identity()
        self.global_stream = global_backbone

        # Local stream backbone
        local_backbone = models.resnet18(weights=weights)
        in_features_l = local_backbone.fc.in_features
        local_backbone.fc = nn.Identity()
        self.local_stream = local_backbone

        # Joint MLP classification head
        total_features = in_features_g + in_features_l  # 512 + 512 = 1024
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(total_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x_global: torch.Tensor, x_local: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with dual visual streams.
        x_global: (B, 3, 224, 224) full light-curve image
        x_local:  (B, 3, 224, 224) cropped red ROI marker image
        """
        feat_global = self.global_stream(x_global)
        feat_local = self.local_stream(x_local)
        fused = torch.cat([feat_global, feat_local], dim=1)
        logits = self.classifier(fused)
        return logits


if __name__ == "__main__":
    model = DualStreamBurstChaserClassifier(num_classes=3, pretrained=False)
    x_g = torch.randn(2, 3, 224, 224)
    x_l = torch.randn(2, 3, 224, 224)
    logits = model(x_g, x_l)
    print(f"Dual-Stream Model initialized: output logits shape = {logits.shape}")

