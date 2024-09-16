import os
import torch
from torchvision import transforms
from PIL import Image
import urllib.request
import matconvnet2tf_torch


def download(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {url} to {filename}...")
        urllib.request.urlretrieve(url, filename)
        print("Download complete.")

def main():
    """Test MatConvNet2PyTorch"""
    # Download the model file if it doesn't exist
    model_url = "http://www.vlfeat.org/matconvnet/models/imagenet-vgg-f.mat"
    model_filename = 'imagenet-vgg-f.mat'
    download(model_url, model_filename)

    # Specify the image file
    image_path = 'image.jpg'
    if not os.path.exists(image_path):
        print(f"Image file '{image_path}' not found.")
        return

    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')

    # Initialize the model
    print(f"Model: {model_filename}")
    model = matconvnet2tf_torch.MatConvNet2PyTorch(model_filename, do_debug_print=True)
    model.eval()

    # Get input size from the model's mean image
    input_size = model.mean.shape[2:]  # (H, W)
    print(f"Expected input size: {input_size}")

    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor()
    ])

    # Apply transformations to the image
    image_tensor = transform(image)  # Shape: [C, H, W]
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension: [1, C, H, W]

    # Run the model on the image
    with torch.no_grad():
        output = model(image_tensor)

    # Get the result and reshape to 1D
    result = output.view(-1)

    # Get top 10 indices by sorting the results
    _, indices = torch.sort(result, descending=True)
    top_indices = indices[:10]

    # Display the top-10 classification results
    for i in top_indices:
        prob = result[i].item() * 100.0
        class_name = model.net['classes'][i]
        print(f"{prob:.2f}% - {class_name}")

if __name__ == "__main__":
    main()