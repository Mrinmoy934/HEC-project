import torch
import torch.nn as nn

class ElephantBehaviourLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        """
        LSTM Model for Elephant Behaviour Detection.
        
        Args:
            input_size (int): Number of input features per time step.
            hidden_size (int): Number of features in the hidden state.
            num_layers (int): Number of recurrent layers.
            num_classes (int): Number of output behaviour classes.
        """
        super(ElephantBehaviourLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Upgrade to GRU for better performance on smaller datasets
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        
        # Heads
        self.fc = nn.Linear(hidden_size, num_classes)
        self.risk_head = nn.Linear(hidden_size, 1) # Output risk score 0-1
        self.alertness_head = nn.Linear(hidden_size, 1) # Output alertness score 0-1 (Calm vs Aggressive)

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input sequence of shape (batch_size, seq_length, input_size).
            
        Returns:
            out (torch.Tensor): Behaviour class logits.
            risk (torch.Tensor): Conflict risk score (sigmoid).
            alertness (torch.Tensor): Alertness score (sigmoid).
        """
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate GRU
        out, _ = self.gru(x, h0)  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        out = out[:, -1, :]
        
        behaviour_out = self.fc(out)
        risk_out = torch.sigmoid(self.risk_head(out))
        alertness_out = torch.sigmoid(self.alertness_head(out))
        
        return behaviour_out, risk_out, alertness_out

if __name__ == "__main__":
    # Example usage
    model = ElephantBehaviourLSTM(input_size=12, hidden_size=64, num_layers=2, num_classes=8)
    dummy_input = torch.randn(1, 30, 12) # Batch=1, Seq=30, Features=12
    beh, risk, alert = model(dummy_input)
    print(f"Behaviour Logits Shape: {beh.shape}")
    print(f"Risk Score: {risk.item()}")
    print(f"Alertness Score: {alert.item()}")
