import torch
import torch.nn as nn
import torchvision.models as models

class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        """
        Initialize the 3D Basic Residual Block (BasicBlock3D).

        Args:
            in_planes (int): Number of input channels.
            planes (int): Number of target output channels.
            stride (int or tuple, optional): Stride for the first 3D convolution.
                Can be a single integer or a tuple of (temporal_stride, spatial_stride, spatial_stride).
                Defaults to 1.
            downsample (nn.Module, optional): Downsample sequential module applied 
                to the shortcut identity connection. Defaults to None.
        """
        super(BasicBlock3D, self).__init__()
        
        # stride can be an integer or a tuple (temporal_stride, spatial_stride, spatial_stride)
        if isinstance(stride, int):
            stride = (1, stride, stride)  # downsampling is applied only in the spatial dimension by default

        # first layer 3D convolution
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=(3, 3, 3), 
                               stride=stride, padding=(1, 1, 1), bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)

        # second layer 3D convolution
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(3, 3, 3), 
                               stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        
        self.downsample = downsample

    def forward(self, x):
        """
        Execute the forward pass of the 3D BasicBlock.

        Args:
            x (torch.Tensor): Input 5D spatio-temporal tensor of shape [B, C_in, T_in, H_in, W_in].

        Returns:
            torch.Tensor: Output 5D spatio-temporal tensor of shape [B, C_out, T_out, H_out, W_out].
        """
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # if downsampling is required (due to channel mismatch or spatial feature map reduction)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet18_3D(nn.Module):
    def __init__(self, num_classes=25, init_strategy='random'):
        """
        3D ResNet-18 backbone implemented from scratch
        
        Args:
            param num_classes: Number of classes (defaults to 25 for miniUCF).
            param init_strategy: Strategy to initialize weights.'Random' (random
                initialization) or 'inflate' (2D weight inflation initialization)
        """
        super(ResNet18_3D, self).__init__()
        self.in_planes = 64

        # conv1: 7x7x7, 64, stride 1 (T), 2 (XY)
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(7, 7, 7), stride=(1, 2, 2), 
                               padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # max pool: 3x3x3, stride 2 (TXY)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))

        # construct 4 residual stages (each stage contains 2 BasicBlock3Ds)
        self.layer1 = self._make_layer(64, 2, stride=1)  # spatial stride=1
        self.layer2 = self._make_layer(128, 2, stride=2) # spatial stride=2
        self.layer3 = self._make_layer(256, 2, stride=2) # spatial stride=2
        self.layer4 = self._make_layer(512, 2, stride=2) # spatial stride=2

        # spatio-temporal average pooling (reduces T, H, W dimensions to 1)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512, num_classes)

        # 1. basic 3D random initialization by default
        self._initialize_weights()
        
        # 2. if inflation strategy is specified, weights are loaded from the official 2D model and inflated
        if init_strategy == 'inflate':
            self.inflate_from_2d()

    def _make_layer(self, planes, num_blocks, stride):
        """
        Build a sequential residual stage containing several 3D basic blocks.

        Args:
            planes (int): Number of target output channels for the stage.
            num_blocks (int): Number of 3D basic blocks inside the stage.
            stride (int): Spatial downsampling stride for the first block.

        Returns:
            nn.Sequential: A sequential module comprising the compiled 3D residual blocks.
        """
        downsample = None
        if stride != 1 or self.in_planes != planes:
            # downsampling is performed only along the spatial dimensions
            downsample = nn.Sequential(
                nn.Conv3d(self.in_planes, planes, kernel_size=(1, 1, 1), 
                          stride=(1, stride, stride), bias=False),
                nn.BatchNorm3d(planes),
            )

        layers = []
        layers.append(BasicBlock3D(self.in_planes, planes, stride, downsample))
        self.in_planes = planes
        for _ in range(1, num_blocks):
            layers.append(BasicBlock3D(self.in_planes, planes))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """
        Standard 3D random initialization (Kaiming Normal).
        """
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def inflate_from_2d(self):
        """
        Inflate PyTorch pre-trained 2D ResNet-18 weights to 3D ResNet-18.
        """
        print("\n=== Starting I3D weight inflation initialization ===")
        # 1. download and load 2D pre-trained weights
        state_dict_2d = models.resnet18(pretrained=True).state_dict()
        state_dict_3d = self.state_dict()
        
        inflated_state_dict = {}
        for key in state_dict_3d.keys():
            # exclude the final classification fc, as the number of classes differs (1000 vs 25)
            if 'fc' in key:
                print(f" -> Skip classification layer weights: {key} (use random initialization)")
                inflated_state_dict[key] = state_dict_3d[key]
                continue
                
            if key in state_dict_2d:
                w_2d = state_dict_2d[key]
                w_3d_shape = state_dict_3d[key].shape
                
                # inflate 2D kernel into 3D kernel
                if len(w_2d.shape) == 4:
                    # determine the size of 3D kernel along the time axis
                    Kt = w_3d_shape[2]
                    # inflation：add a dimension at dim=2 (time axis), replicate it Kt times, and divide by Kt
                    w_3d = w_2d.unsqueeze(2).repeat(1, 1, Kt, 1, 1) / Kt
                    inflated_state_dict[key] = w_3d
                    print(f" -> Successful inflated convolution layer: {key} | Shape change: {list(w_2d.shape)} -> {list(w_3d.shape)}")
                else:
                    # for one-dimensional parameters (e.g., scale/bias/mean/var of BN), perform a lossless copy
                    inflated_state_dict[key] = w_2d
                    print(f" -> Directly copy BN/bias parameters: {key} | Shape: {list(w_2d.shape)}")
            else:
                print(f" -> Warning: 2D correspondence weights not found, use random initialization.: {key}")
                inflated_state_dict[key] = state_dict_3d[key]
                
        # load the complete set of weights after infation and combination
        self.load_state_dict(inflated_state_dict)
        print("=== I3D Weight inflation initialization completed successfully. ===\n")

    def forward(self, x):
        """Execute the forward pass of the 3D ResNet-18.

        Args:
            x (torch.Tensor): Input 5D spatio-temporal tensor of shape [B, C_in, T_in, H_in, W_in].

        Returns:
            torch.Tensor: Predicted logits tensor of shape [B, num_classes].
        """
        # first layer convolution, BN, ReLU, and pooling
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        # 4 residual stages
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # pooling and flattening
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        
        # classifier output
        out = self.fc(out)
        return out