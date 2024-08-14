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

# Load a ResNet model pre-trained on CIFAR-10
model = torchvision.models.resnet18(pretrained=False, num_classes=10)
model.load_state_dict(torch.hub.load_state_dict_from_url('https://github.com/huyvnphan/PyTorch_CIFAR10/raw/main/pretrained_models/resnet18_cifar10.pth'))

model.eval()  # Set the model to evaluation mode

# Load MNIST data
mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

data_path = "data/mnist"
mnist_train = torchvision.datasets.MNIST(root=data_path, train=True, download=True, transform=mnist_transform)
mnist_test = torchvision.datasets.MNIST(root=data_path, train=False, download=True, transform=mnist_transform)

# Define a simple CNN model (LeNet-5-like architecture)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = torch.relu(torch.max_pool2d(self.conv1(x), 2))
        x = torch.relu(torch.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)

# Instantiate and load the pre-trained weights
model = SimpleCNN()
model.load_state_dict(torch.hub.load_state_dict_from_url('https://github.com/ArashHosseini/Simple_CNN_MNIST/raw/main/model_mnist.pth'))

model.eval()  # Set the model to evaluation mode