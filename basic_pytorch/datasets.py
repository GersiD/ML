import torch
import torchvision
from torchvision import transforms
import os

assert os.path.exists('/mnt/2tb-drive/data'), "Data directory does not exist. Please check the path."
transform = torchvision.transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
traincifar10 = torchvision.datasets.CIFAR10('/mnt/2tb-drive/data/cifar10', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(traincifar10, batch_size=32, shuffle=True, num_workers=16)
testcifar10 = torchvision.datasets.CIFAR10('/mnt/2tb-drive/data/cifar10', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testcifar10, batch_size=32, shuffle=False, num_workers=16)


class CNN(torch.nn.Module):
    """
    A simple CNN model for CIFAR-10 classification.

    Inputs will be 32x32 RGB images, and the model will output class probabilities for 10 classes.
    """
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = torch.nn.Linear(64 * 8 * 8, 512)
        self.fc2 = torch.nn.Linear(512, 10)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)  # Flatten the tensor
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
# Example usage
epochs = 5
model = CNN()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model.train()  # Set the model to training mode
for epoch in range(epochs):
    for i, (images, labels) in enumerate(trainloader):
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

        if (i + 1) % 100 == 0:  # Print every 100 batches
            print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(trainloader)}], Loss: {loss.item():.4f}')
# Evaluate the model
model.eval()  # Set the model to evaluation mode
correct = 0
total = 0
with torch.no_grad():
    for images, labels in testloader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f'Accuracy of the model on the test images: {100 * correct / total:.2f}%') # We get around 73% accuracy on the test set
