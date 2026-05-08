"""
Loss functions for FIB-SEM multi-class segmentation.

- DiceLoss:            Weighted soft Dice, numerically stable
- CombinedDiceCELoss:  50% Dice + 50% Cross-Entropy (configurable)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Weighted Soft Dice Loss — numerically stable.

    Ensures loss is always in [0, 1], with smooth gradients.

    Args:
        num_classes:   Number of output classes
        class_weights: Per-class weight list (length == num_classes)
        smooth:        Laplace smoothing epsilon (default 1e-5)
    """

    def __init__(self, num_classes: int, class_weights=None, smooth: float = 1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

        if class_weights is None:
            self.register_buffer('class_weights', torch.ones(num_classes))
        else:
            self.register_buffer('class_weights',
                                 torch.tensor(class_weights, dtype=torch.float32))

    def forward(self, predictions, targets):
        """
        Args:
            predictions: (B, C, Z, Y, X) logits
            targets:     (B, Z, Y, X) class indices

        Returns:
            Scalar loss in [0, 1]
        """
        probs = F.softmax(predictions, dim=1)
        B, C = probs.shape[0], probs.shape[1]

        probs = probs.view(B, C, -1)           # (B, C, N)
        targets = targets.view(B, -1)          # (B, N)

        targets_onehot = torch.zeros(B, C, targets.shape[1], device=targets.device)
        targets_onehot.scatter_(1, targets.unsqueeze(1), 1.0)

        intersection = torch.sum(probs * targets_onehot, dim=2)    # (B, C)
        cardinality  = torch.sum(probs + targets_onehot, dim=2)    # (B, C)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        dice = torch.clamp(dice, min=0.0, max=1.0)

        weighted_dice = dice * self.class_weights.unsqueeze(0)     # (B, C)
        loss = 1.0 - (weighted_dice.sum() / (self.class_weights.sum() * B))

        return loss


class CombinedDiceCELoss(nn.Module):
    """
    Combined Weighted Dice + Cross-Entropy Loss.

    Args:
        num_classes:   Number of output classes
        class_weights: Per-class weight list
        dice_weight:   Contribution of Dice loss (default 0.5)
        ce_weight:     Contribution of CE loss   (default 0.5)
        smooth:        Dice smoothing epsilon
    """

    def __init__(self, num_classes: int, class_weights=None,
                 dice_weight: float = 0.5, ce_weight: float = 0.5,
                 smooth: float = 1e-5):
        super().__init__()

        self.dice_loss = DiceLoss(num_classes, class_weights, smooth)

        ce_weights = (torch.tensor(class_weights, dtype=torch.float32)
                      if class_weights is not None else None)
        self.ce_loss = nn.CrossEntropyLoss(weight=ce_weights, reduction='mean')

        self.dice_weight = dice_weight
        self.ce_weight   = ce_weight

        print(f"  Dice weight:          {dice_weight:.0%}")
        print(f"  Cross-Entropy weight: {ce_weight:.0%}")
        if class_weights is not None:
            print(f"  Class weights:        {class_weights}")
        print(f"  Smoothing epsilon:    {smooth}")

    def forward(self, predictions, targets):
        """
        Args:
            predictions: (B, C, Z, Y, X) logits
            targets:     (B, Z, Y, X) class indices

        Returns:
            Scalar loss (always >= 0)
        """
        dice = torch.clamp(self.dice_loss(predictions, targets), min=0.0)
        ce   = torch.clamp(self.ce_loss(predictions, targets),   min=0.0)

        return torch.clamp(self.dice_weight * dice + self.ce_weight * ce, min=0.0)
