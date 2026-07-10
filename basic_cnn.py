import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Find device for nn
device = None

if torch.cuda.is_available():
    torch.device("cuda")

else:
    torch.device("cpu")

print(f"Device: {device}")

# Prepare the transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) # Typical MNISt scaled avg and std
])

# Load the dataset, 1 for training, 1 for proper testing
trainDataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
testDataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

# Get dataloaders
trainLoader = DataLoader(trainDataset, batch_size=64, shuffle=True)
testLoader = DataLoader(testDataset, batch_size=1000, shuffle=False)

sample_images = trainDataset.data.numpy()  # shape (60000, 28, 28), dtype uint8
print("Pixel value range:", sample_images.min(), "-", sample_images.max())
print("Pixel value mean:", np.mean(sample_images))

# Make the class for the cnn
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()

        # 2 layer nn
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1) # 8 reccomended for cpu, padding of 1 so it gets back to 28x28 using 3x3 kernel
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)

        # ReLU, maxpooling and dropout and probably some form of optimisation
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # Keep in mind this will halve length and width
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)

        # My layers
        self.layer1 = nn.Linear(784, 128)
        self.layer2 = nn.Linear(128, 10)

        def forward(self, x):
            # Standard process is conv then relu then pool
            x = self.conv1(x)
            x = self.relu(x)
            x = self.pool(x)

            x = self.conv2(x)
            x = self.relu(x)
            x = self.pool(x)

            x = x.view(x.size(0), -1) # Flattening to ([batch size, flattened dimensions])
            
            x = self.relu(self.layer1(x))
            x = self.dropout(x)
            x = self.layer2(x)
            return x

model = CNNModel().to(device)
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def train():
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)


# Find loss and other stuff

