import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F

class MatConvNet2PyTorch(nn.Module):
    """Reads matconvnet file and creates a PyTorch model."""
    def __init__(self, data_path, ignore=[], do_debug_print=False):
        super(MatConvNet2PyTorch, self).__init__()
        data = scipy.io.loadmat(data_path, struct_as_record=False, squeeze_me=True)
        layers = data['layers']
        self.net = {}
        try:
            self.mean = np.array(data['meta'].normalization.averageImage, ndmin=4)
            self.net['classes'] = data['meta'].classes.description
        except KeyError:
            self.mean = np.array(data['normalization'].averageImage, ndmin=4)

        self.mean = torch.from_numpy(self.mean).float()
        self.do_debug_print = do_debug_print
        self.layers = nn.ModuleList()

        self.layer_types = {
            'conv': self._conv_layer,
            'relu': self._relu_layer,
            'pool': self._pool_layer,
            'lrn': self._lrn_layer,
            'normalize': self._lrn_layer,
            'softmax': self._softmax_layer,
            'dagnn.Conv': self._conv_layer,
            'dagnn.ReLU': self._relu_layer,
            'dagnn.Pooling': self._pool_layer,
            'dagnn.SoftMax': self._softmax_layer,
        }

        for i, layer in enumerate(layers):
            if layer.name not in ignore:
                layer_type = layer.type
                if layer_type in self.layer_types:
                    layer_instance = self.layer_types[layer_type](layer)
                    self.layers.append(layer_instance)
                else:
                    print(f"Unknown layer type: {layer_type}")

    def forward(self, x):
        x = x - self.mean.to(x.device)
        for layer in self.layers:
            x = layer(x)
        return x

    def _conv_layer(self, layer):
        weights, biases = layer.weights
        biases = biases.reshape(-1)
        weights = np.array(weights)
        biases = np.array(biases)

        if len(weights.shape) == 2:
            # Fully connected layer
            in_features, out_features = weights.shape
            fc_layer = FullyConnectedLayer(in_features, out_features, weights.T, biases)
            return fc_layer
        else:
            # Convolutional layer
            weights = weights.transpose(3, 2, 0, 1)
            out_channels, in_channels, kernel_height, kernel_width = weights.shape

            pad = layer.pad
            stride = layer.stride

            if (pad != [0, 0, 0, 0]).any():
                padding = (pad[2], pad[3], pad[0], pad[1])  # (left, right, top, bottom)
                padding_layer = nn.ZeroPad2d(padding)
            else:
                padding_layer = None

            stride = tuple(stride)

            conv = nn.Conv2d(in_channels, out_channels,
                             kernel_size=(kernel_height, kernel_width),
                             stride=stride, padding=0, bias=True)
            conv.weight.data = torch.from_numpy(weights).float()
            conv.bias.data = torch.from_numpy(biases).float()

            layers = []
            if padding_layer is not None:
                layers.append(padding_layer)
            layers.append(conv)

            return nn.Sequential(*layers)

    def _relu_layer(self, layer):
        return nn.ReLU(inplace=True)

    def _pool_layer(self, layer):
        pad = layer.pad
        stride = layer.stride
        pool_size = layer.pool

        if (pad != [0, 0, 0, 0]).any():
            padding = (pad[2], pad[3], pad[0], pad[1])  # (left, right, top, bottom)
            padding_layer = nn.ZeroPad2d(padding)
        else:
            padding_layer = None

        stride = tuple(stride)
        kernel_size = tuple(pool_size)

        pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=0)

        layers = []
        if padding_layer is not None:
            layers.append(padding_layer)
        layers.append(pool)

        return nn.Sequential(*layers)

    def _lrn_layer(self, layer):
        n = layer.param[0]
        bias = layer.param[1]
        alpha = layer.param[2]
        beta = layer.param[3]

        lrn = nn.LocalResponseNorm(size=n, alpha=alpha, beta=beta, k=bias)

        return lrn

    def _softmax_layer(self, layer):
        return nn.Softmax(dim=1)

class FullyConnectedLayer(nn.Module):
    def __init__(self, in_features, out_features, weights, biases):
        super(FullyConnectedLayer, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.fc.weight.data = torch.from_numpy(weights).float()
        self.fc.bias.data = torch.from_numpy(biases).float()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
