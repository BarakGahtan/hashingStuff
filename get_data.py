import os

import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn

# Define the data directory
data_path = "data/"

# Download and load the MNIST dataset
mnist_transform = transforms.Compose([transforms.ToTensor()])

mnist_train = torchvision.datasets.MNIST(root=os.path.join(data_path, "mnist"), train=True, download=True, transform=mnist_transform)
mnist_test = torchvision.datasets.MNIST(root=os.path.join(data_path, "mnist"), train=False, download=True, transform=mnist_transform)

# Download and load the CIFAR-10 dataset
cifar10_transform = transforms.Compose([transforms.ToTensor()])

cifar10_train = torchvision.datasets.CIFAR10(root=os.path.join(data_path, "cifar10"), train=True, download=True, transform=cifar10_transform)
cifar10_test = torchvision.datasets.CIFAR10(root=os.path.join(data_path, "cifar10"), train=False, download=True, transform=cifar10_transform)

# Example: Accessing a single image and label
mnist_image, mnist_label = mnist_train[0]
cifar10_image, cifar10_label = cifar10_train[0]

print(f"MNIST image shape: {mnist_image.shape}, label: {mnist_label}")
print(f"CIFAR-10 image shape: {cifar10_image.shape}, label: {cifar10_label}")

# Define the transform for CIFAR-10
cifar10_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Load the CIFAR-10 dataset
data_path = "data/cifar10"
cifar10_train = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=cifar10_transform)
cifar10_test = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=cifar10_transform)

# Load a ResNet-18 model pre-trained on ImageNet and modify it for CIFAR-10
model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 10)  # Modify the output layer to match CIFAR-10 classes

model.eval()  # Set the model to evaluation mode

# Load MNIST data
mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

