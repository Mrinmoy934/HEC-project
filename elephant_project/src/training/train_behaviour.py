import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from behaviour_dataset import BehaviourDataset
import sys
import os

# Add models directory to path to import LSTM
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, '../models')
sys.path.append(models_dir)
from lstm_model import ElephantBehaviourLSTM

import argparse

    # ... (imports remain the same, but we'll clean them up in the full file if needed, here just replacing the main block and adding argparse)

def train_behaviour_model(csv_path, epochs=100, batch_size=32):
    """
    Train the LSTM Behaviour Model.
    """
    # Hyperparameters
    input_size = 10 # Number of features in CSV
    hidden_size = 128
    num_layers = 2
    num_classes = 8 # As per plan
    learning_rate = 0.001
    
    # Device config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dataset & Loader
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")
        
    dataset = BehaviourDataset(csv_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Model
    model = ElephantBehaviourLSTM(input_size, hidden_size, num_layers, num_classes).to(device)
    
    # Loss and Optimizer
    criterion_cls = nn.CrossEntropyLoss()
    criterion_risk = nn.MSELoss() 
    criterion_alert = nn.MSELoss() # Binary classification treated as regression 0-1 for simplicity, or BCELoss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training Loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for i, (features, labels, risks, alerts) in enumerate(dataloader):
            features = features.to(device)
            labels = labels.to(device)
            risks = risks.to(device).unsqueeze(1) # Match shape (batch, 1)
            alerts = alerts.to(device).unsqueeze(1) # Match shape (batch, 1)
            
            # Forward
            outputs, risk_preds, alert_preds = model(features)
            
            loss_cls = criterion_cls(outputs, labels)
            loss_risk = criterion_risk(risk_preds, risks)
            loss_alert = criterion_alert(alert_preds, alerts)
            
            loss = loss_cls + loss_risk + loss_alert
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}')
        
    # Save model
    save_path = os.path.join(models_dir, 'lstm/behaviour_lstm.pth')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Training finished. Model saved to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Elephant Behaviour Model')
    parser.add_argument('--csv', type=str, default='behaviour_dataset_extracted.csv', help='Path to dataset CSV')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    
    args = parser.parse_args()
    
    if os.path.exists(args.csv):
        train_behaviour_model(args.csv, args.epochs, args.batch_size)
    else:
        print(f"Dataset not found at {args.csv}. Please provide a valid path using --csv")
