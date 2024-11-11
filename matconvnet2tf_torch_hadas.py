import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F



class VGGF(nn.Module):


        def _init_(self):
            super(VGGF, self)._init_(ignore=['fc8', 'prob'])

            self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=1)
            self.conv2 = nn.Conv2d(64, 256, kernel_size=5, padding=2)
            self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
            self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
            self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

            self.fc6 = nn.Linear(256 * 6 * 6, 4096)
            self.fc7 = nn.Linear(4096, 4096)
            self.fc8 = nn.Linear(4096, 1000)

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