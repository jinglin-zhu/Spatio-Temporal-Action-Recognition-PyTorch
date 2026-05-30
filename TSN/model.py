import torch
import torch.nn as nn
import torchvision.models as models

class TSN(nn.Module):
    def __init__(self, num_classes=25, modality='RGB', num_segments=4, init_strategy='imagenet'):
        """
        Initialize the Temporal Segment Network (TSN) model.

        Args:
            num_classes (int, optional): Number of actions to classify. Defaults to 25.
            modality (str, optional): Input modality. Options: 'RGB' or 'Flow'. Defaults to 'RGB'.
            num_segments (int, optional): Number of segments (K) to aggregate. Defaults to 4.
            init_strategy (str, optional): Initialization method. Options: 'imagenet' (ImageNet 
                pretrained) or 'random' (train from scratch). Defaults to 'imagenet'.
        """
        super(TSN, self).__init__()
        self.num_classes = num_classes
        self.modality = modality
        self.num_segments = num_segments
        
        # Load the standard ResNet-18 base model
        if init_strategy == 'imagenet':
            # Use torchvision weights enum as the 'pretrained' parameter is deprecated
            from torchvision.models import ResNet18_Weights
            self.base_model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        else:
            self.base_model = models.resnet18(weights=None)
            
        # Replace the final fully connected layer to match the target class count (25 classes)
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(in_features, num_classes)
        
        # Initialize the newly replaced fully connected layer
        nn.init.normal_(self.base_model.fc.weight, 0, 0.01)
        nn.init.constant_(self.base_model.fc.bias, 0)
        
        # Reconstruct the first convolutional layer if the modality is Optical Flow
        if self.modality == 'Flow':
            self._reconstruct_first_layer(init_strategy)

    def _reconstruct_first_layer(self, init_strategy):
        """
        Reconstruct the first convolutional layer to accept 10-channel flow inputs.

        Args:
            init_strategy (str): The weight initialization strategy ('imagenet' or 'random').
        """
        # Create a new Conv2d layer configured for 10 channels
        # Standard conv1 in ResNet-18 has 64 output channels, kernel_size 7, stride 2, padding 3, bias False
        new_conv = nn.Conv2d(10, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        if init_strategy == 'imagenet':
            # Cross-modality pre-training: average the pretrained 3-channel weights and copy 10 times
            old_weight = self.base_model.conv1.weight.data  # Shape: [64, 3, 7, 7]
            mean_weight = old_weight.mean(dim=1, keepdim=True)  # Shape: [64, 1, 7, 7]
            new_weight = mean_weight.repeat(1, 10, 1, 1)  # Shape: [64, 10, 7, 7]
            new_conv.weight.data = new_weight
        else:
            # Standard Kaiming Normal initialization for random strategy
            nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
            
        self.base_model.conv1 = new_conv

    def forward(self, x):
        """
        Forward pass of the Temporal Segment Network.

        Args:
            x (torch.Tensor): Input video tensor of shape [Batch, num_segments, Channels, H, W].
                For RGB, Channels = 3. For Flow, Channels = 10.

        Returns:
            torch.Tensor: Video-level prediction logits of shape [Batch, num_classes].
        """
        batch_size = x.size(0)
        channels = x.size(2)
        height = x.size(3)
        width = x.size(4)
        
        # Reshape input to merge Batch and Segment dimensions: [Batch * num_segments, Channels, H, W]
        # This allows parallel 2D feature extraction across all segments of the batch
        x = x.view(-1, channels, height, width)
        
        # Extract segment-level class logits
        logits = self.base_model(x)  # Shape: [Batch * num_segments, num_classes]
        
        # Reshape back to separate Batch and Segment dimensions
        logits = logits.view(batch_size, self.num_segments, self.num_classes)
        
        # Segmental Consensus: Aggregate segment predictions using average pooling
        consensus_logits = torch.mean(logits, dim=1)  # Shape: [Batch, num_classes]
        
        return consensus_logits