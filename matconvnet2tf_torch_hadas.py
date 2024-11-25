import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F



class VGGF(nn.Module):
    def __init__(self):
        super(VGGF, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=1)
        self.conv2 = nn.Conv2d(64, 256, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.fc6 = nn.Linear(256 * 6 * 6, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 24)
        # self.fc8.weight.data.normal_(0, 0.01)
        # self.fc8.bias.data.normal_(0, 0.01)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=3, stride=2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=3, stride=2)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.max_pool2d(x, kernel_size=3, stride=2)

        x = x.view(x.size(0), -1)  # Flatten the tensor

        x = F.relu(self.fc6(x))
        x = F.dropout(x, 0.5)

        x = F.relu(self.fc7(x))
        x = F.dropout(x, 0.5)

        x = self.fc8(x)
        return x


def load_matconvnet_weights(model):
    """
    Load weights from a MatConvNet .mat file into a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model to load weights into.

    Returns:
        nn.Module: Model with loaded weights.
    """
    import scipy.io
    import torch
    from torch import nn

    # Load .mat file
    mat_data = scipy.io.loadmat('data/imagenet-matconvnet-vgg-f.mat')
    layers = mat_data['layers'][0]  # Extract layers

    # Layer mapping from MatConvNet to PyTorch
    layer_map = {name: layer for name, layer in model.named_modules()}

    for i, layer_data in enumerate(layers):
        layer_name = layer_data[0][0][1][0]  # Layer name
        layer_weights = layer_data[0][0][2][0]  # Weights and biases

        if layer_name in layer_map:
            layer = layer_map[layer_name]

            if isinstance(layer, nn.Conv2d):
                # Transpose weights from [H, W, C_in, C_out] to [C_out, C_in, H, W]
                weights = torch.tensor(layer_weights[0], dtype=torch.float32)
                weights = weights.permute(3, 2, 0, 1)  # Transpose dimensions
                biases = torch.tensor(layer_weights[1].flatten(), dtype=torch.float32)  # Flatten biases
                if layer.weight.shape == weights.shape:
                    layer.weight.data = weights
                if layer.bias.shape == biases.shape:
                    layer.bias.data = biases


            elif isinstance(layer, nn.Linear):
                if layer_name == "fc8":
                    layer.weight.data.normal_(0, 0.01)
                    layer.bias.data.normal_(0, 0.01)
                else:
                    weights = torch.tensor(layer_weights[0], dtype=torch.float32)
                    weights = weights.permute(3, 2, 0, 1)  # Transpose dimensions
                    biases = torch.tensor(layer_weights[1].flatten(), dtype=torch.float32)  # Flatten biases
                    # Flatten the weights: [output_dim, input_channels, height, width] -> [output_dim, input_dim]
                    weights = weights.reshape(weights.size(0),-1)  # Combine input_channels, height, width into a single dimension
                    if layer.weight.shape == weights.shape:  # PyTorch expects [input_dim, output_dim]
                        layer.weight.data = weights  # Transpose to match PyTorch
                    if layer.bias.shape == biases.shape:
                        layer.bias.data = biases



    return model
def load_weights(model, data):
    # Assuming data['layers'] contains weights/biases in the order of layers
    # Access each conv or fc layer and set weights and biases
    layer_idx = 0  # Initialize to keep track of layers in .mat file

    for name, layer in model.named_children():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            # Load weights and biases for this layer
            weight = data['layers'][0][layer_idx][0][0][0][0]
            bias = data['layers'][0][layer_idx][0][0][0][1].reshape(-1)

            # Set weights and biases
            layer.weight.data = torch.tensor(weight).permute(3, 2, 0, 1)  # For Conv2D
            layer.bias.data = torch.tensor(bias)

            layer_idx += 1
