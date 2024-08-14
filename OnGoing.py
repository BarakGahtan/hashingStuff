#TODO:
# 1. Donload CIFAR and MNIST dataset.
# 2. Replicate some results (Table 1) CNN model on CIFAR dataset
# 3. Similar to other deep hashing methods we use raw image pixels as input.
    # Following [42, 49, 23], for the spherical embedding we adopt VGG-F [6] pre-trained on ImageNet.
    # We replace the last layer with our own, initialized with normal distribution.
    # The output layer doesn’t have activation function and the number of outputs matches the needed number of bits - B.
    # The input layer of VGG-F is 224x224, so we crop and resize images of the NUS WIDE dataset and upsample images of the CIFAR-10 dataset to match the input size.

# 4. Download a pretrained moodel for MNIST and CIFAR and download both datasets.