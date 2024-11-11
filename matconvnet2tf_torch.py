import numpy as np
import scipy.io
import torch
import torch.nn as nn


def _check_keys(dict):
    """Check if entries in dictionary are mat-objects. If yes, convert them to nested dictionaries."""
    for key in dict:
        if isinstance(dict[key], scipy.io.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict

def _todict(matobj):
    """A recursive function which converts matobjects to nested dictionaries."""
    dict = {}
    for strg in matobj._fieldnames:
        elem = getattr(matobj, strg)
        if isinstance(elem, scipy.io.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict
class MatConvNet2PyTorch(nn.Module):
    def __init__(self, data_path, ignore=[], do_debug_print=False):
        super(MatConvNet2PyTorch, self).__init__()
        data = scipy.io.loadmat(data_path, struct_as_record=False, squeeze_me=True)
        data = _check_keys(data)  # Convert `mat_struct` objects to dictionaries

        layers = data['layers']
        self.net = {}
        try:
            self.mean = np.array(data['meta']['normalization']['averageImage'], ndmin=4)
            self.net['classes'] = data['meta']['classes']['description']
        except KeyError:
            self.mean = np.array(data['normalization']['averageImage'], ndmin=4)

        self.mean = torch.from_numpy(self.mean).float()
        self.do_debug_print = do_debug_print
        self.layers = nn.ModuleList()

        self.layer_types = {
            'conv': self._conv_layer,
            'relu': self._relu_layer,
            'pool': self._pool_layer,
            'lrn': self._lrn_layer,
            'softmax': self._softmax_layer,
        }

        for i, layer in enumerate(layers):
            if getattr(layer, 'name', None) not in ignore:
                layer_type = getattr(layer, 'type', '')
                if layer_type in self.layer_types:
                    layer_instance = self.layer_types[layer_type](layer)
                    self.layers.append(layer_instance)
                else:
                    if self.do_debug_print:
                        print(f"Unknown layer type: {layer_type}")

    def forward(self, x):
        x = x - self.mean.to(x.device)
        for layer in self.layers:
            x = layer(x)
        return x

    def _conv_layer(self, layer):
        weights, biases = layer.weights
        weights = torch.from_numpy(weights.transpose(3, 2, 0, 1)).float()  # Convert to PyTorch format
        biases = torch.from_numpy(biases.reshape(-1)).float()

        stride = tuple(layer.stride) if hasattr(layer, 'stride') else (1, 1)
        padding = tuple(layer.pad) if hasattr(layer, 'pad') and isinstance(layer.pad, (list, tuple)) else (0, 0)

        conv = nn.Conv2d(weights.shape[1], weights.shape[0],
                         kernel_size=(weights.shape[2], weights.shape[3]),
                         stride=stride, padding=0, bias=True)
        conv.weight.data = weights
        conv.bias.data = biases
        return conv

    def _relu_layer(self, layer):
        return nn.ReLU(inplace=True)

    def _pool_layer(self, layer):
        pool_size = getattr(layer, 'pool', [2, 2])
        stride = getattr(layer, 'stride', pool_size)
        return nn.MaxPool2d(kernel_size=tuple(pool_size), stride=tuple(stride))

    def _lrn_layer(self, layer):
        params = getattr(layer, 'param', [5, 0.0001, 0.75, 1])
        return nn.LocalResponseNorm(size=params[0], alpha=params[2], beta=params[3], k=params[1])

    def _softmax_layer(self, layer):
        return nn.Softmax(dim=1)
