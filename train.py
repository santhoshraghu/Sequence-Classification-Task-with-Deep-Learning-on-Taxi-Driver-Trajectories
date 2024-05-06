import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from model import TaxiDriverClassifier
from extract_feature import load_data, preprocess_data

if torch.cuda.is_available():
  device = torch.device("cuda:0")
  print("GPU")
else:
  device = torch.device("cpu")
  print("CPU")

class TaxiDriverDataset(Dataset):
    """
    Custom dataset class for Taxi Driver Classification.
    Handles loading and preparing data for the model
    """
    def __init__(self, X, y, device):
        self.X = torch.tensor(X, dtype=torch.float32).to(device)
        self.y = torch.tensor(y, dtype=torch.long).to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train(model, optimizer, criterion, train_loader, device):
    """
    Function to handle the training of the model.
    Iterates over the training dataset and updates model parameters.
    """
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss /= len(train_loader)
    train_acc = 100. * correct / total
    return train_loss, train_acc


def train_model():
    """
    Main function to initiate the model training process.
    Includes loading data, setting up the model, optimizer, and criterion,
    and executing the training and validation loops.
    """

    # Get the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("GPU" if torch.cuda.is_available() else "CPU")

    # Load and preprocess data
    X_train, _, y_train, _ = load_data("C:\\Users\\santh\\Desktop\\BDA-P2\\data_5drivers\\*.csv")

    # Create dataset and dataloaders for training data
    train_dataset = TaxiDriverDataset(X_train, y_train, device)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Initialize the model
    model = TaxiDriverClassifier(input_dim=X_train.shape[2], hidden_dim=128, num_layers=2, output_dim=len(np.unique(y_train))).to(device)

    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Lists to store training statistics
    train_losses = []
    train_accuracies = []

    # Training loop
    for epoch in range(15):  # Example: 10 epochs
        train_loss, train_acc = train(model, optimizer, criterion, train_loader, device)
        print(f"Epoch {epoch+1}/{15}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        
        # Store training statistics
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

    # Save the trained model in both .pth and .pt formats
    torch.save(model.state_dict(), "trained_model.pth")
    torch.save(model, "trained_model.pt")
    print("Model saved successfully.")

    # Plotting the graph
    epochs = range(1, 16)  # Number of epochs
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss vs Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Accuracy vs Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_model()
