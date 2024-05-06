import torch.nn as nn

class TaxiDriverClassifier(nn.Module):
    """
    RNN model using LSTM for taxi driver classification.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(TaxiDriverClassifier, self).__init__()

        # Define LSTM layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Define fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Forward pass of the model.
        """
        # Forward pass through LSTM layers
        lstm_out, _ = self.lstm(x)

        # Get the output of the last time step
        out = self.fc(lstm_out[:, -1, :])

        return out
