import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from model import TaxiDriverClassifier
from train import TaxiDriverDataset
from extract_feature import load_data

def test(model, test_loader, device):
    """
    Test the model performance on the test set.
    """
    model.eval()
    test_loss = 0
    test_correct = 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            test_correct += predicted.eq(labels).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = 100. * test_correct / len(test_loader.dataset)
    return test_loss, test_acc

def test_model():
    """
    Main function to initiate the model testing process.
    Includes loading test data, setting up the model and test loader,
    and executing the testing function.
    """
    # Get the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load test data
    _, X_test, _, y_test = load_data("C:\\Users\\santh\\Desktop\\BDA-P2\\data_5drivers\\*.csv")

    # Create dataset and dataloader for test data
    test_dataset = TaxiDriverDataset(X_test, y_test, device)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Initialize the model
    model = TaxiDriverClassifier(input_dim=X_test.shape[2], hidden_dim=128, num_layers=2, output_dim=len(np.unique(y_test))).to(device)

    # Load the trained model weights
    model.load_state_dict(torch.load("trained_model.pth"))

    # Test the model
    test_loss, test_acc = test(model, test_loader, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

if __name__ == "__main__":
    test_model()
