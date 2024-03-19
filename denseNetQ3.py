import torch
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from torch import nn, optim
from tqdm import tqdm

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize pixel values
])
trainset = MNIST(root='./data', train=True, download=True, transform=transform)
testset = MNIST(root='./data', train=False, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

# Define the model
class DenseNet(nn.Module):
    def __init__(self, input_size, hidden_layers, neurons_per_layer, output_size, activation):
        super(DenseNet, self).__init__()
        layers = []
        layers.append(nn.Flatten())  # Flatten input image
        for _ in range(hidden_layers):
            layers.append(nn.Linear(input_size, neurons_per_layer))  # Add dense layer
            if activation == 'relu':
                layers.append(nn.ReLU())  # Add ReLU activation
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())  # Add Sigmoid activation
            input_size = neurons_per_layer  # Update input size for next layer
        layers.append(nn.Linear(neurons_per_layer, output_size))  # Output layer
        layers.append(nn.LogSoftmax(dim=1))  # LogSoftmax for classification
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def train_model(model, trainloader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in tqdm(trainloader, desc=f'Epoch {epoch+1}/{epochs}'):
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1} loss: {running_loss/len(trainloader)}')

def evaluate_model(model, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            output = model(images)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print('Test accuracy:', accuracy)

# Hyperparameters
input_size = 28 * 28  # Input image size
hidden_layers = 2  # Number of hidden layers
neurons_per_layer = 128  # Neurons per hidden layer
output_size = 10  # Number of output classes
activation = 'relu'  # Activation function ('relu' or 'sigmoid')

# Create a PyTorch model
model = DenseNet(input_size, hidden_layers, neurons_per_layer, output_size, activation)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 10
train_model(model, trainloader, criterion, optimizer, epochs)

# Evaluate the model
evaluate_model(model, testloader)
