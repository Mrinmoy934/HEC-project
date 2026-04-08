import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class BehaviourDataset(Dataset):
    def __init__(self, csv_file, seq_length=30):
        """
        Custom Dataset for Elephant Behaviour.
        
        Args:
            csv_file (str): Path to the CSV file with annotations.
            seq_length (int): Length of the sequence for LSTM.
        """
        self.data = pd.read_csv(csv_file)
        self.seq_length = seq_length
        
        # Feature columns
        self.feature_cols = [
            'posture_class', 'x1', 'y1', 'x2', 'y2', 
            'move_dx', 'move_dy', 'trunk_angle', 'ear_freq', 'tail_freq'
        ]
        # Label columns
        self.label_col = 'behaviour_label'
        self.risk_col = 'conflict_risk'
        self.alertness_col = 'alertness_label' # New column
        
        # Mapping labels to integers (example)
        self.label_map = {label: i for i, label in enumerate(self.data[self.label_col].unique())}
        self.risk_map = {'Low': 0.0, 'Medium': 0.5, 'High': 1.0}
        self.alertness_map = {'Calm': 0.0, 'Aggressive': 1.0}

        self.sequences = self._create_sequences()

    def _create_sequences(self):
        sequences = []
        # Group by elephant_id to ensure sequences are from the same individual
        grouped = self.data.groupby('elephant_id')
        
        for _, group in grouped:
            group = group.sort_values('frame_id')
            values = group[self.feature_cols].values
            labels = group[self.label_col].values
            risks = group[self.risk_col].values
            # Handle missing alertness column gracefully if needed, but assuming it exists for now
            if self.alertness_col in group.columns:
                alerts = group[self.alertness_col].values
            else:
                # Fallback: Derive from behaviour if possible, or default to Calm
                alerts = ['Calm'] * len(group) 
            
            num_frames = len(group)
            if num_frames < self.seq_length:
                continue
                
            for i in range(num_frames - self.seq_length):
                seq_features = values[i : i + self.seq_length]
                # Label is the label of the last frame in the sequence
                target_label = self.label_map[labels[i + self.seq_length - 1]]
                target_risk = self.risk_map[risks[i + self.seq_length - 1]]
                target_alert = self.alertness_map.get(alerts[i + self.seq_length - 1], 0.0)
                
                sequences.append((seq_features, target_label, target_risk, target_alert))
                
        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        features, label, risk, alert = self.sequences[idx]
        
        # Data Augmentation: Add Gaussian noise to features
        if self.data is not None: # Check if training mode (simplification)
            noise = np.random.normal(0, 0.01, features.shape)
            features = features + noise
            
        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
            torch.tensor(risk, dtype=torch.float32),
            torch.tensor(alert, dtype=torch.float32)
        )

if __name__ == "__main__":
    # Dummy CSV creation for testing
    import os
    if not os.path.exists('dummy_data.csv'):
        df = pd.DataFrame({
            'frame_id': range(100),
            'elephant_id': [1]*100,
            'posture_class': [0]*100,
            'x1': [100]*100, 'y1': [100]*100, 'x2': [200]*100, 'y2': [200]*100,
            'move_dx': [0.1]*100, 'move_dy': [0.1]*100,
            'trunk_angle': [45]*100, 'ear_freq': [0.5]*100, 'tail_freq': [0.2]*100,
            'behaviour_label': ['Calm']*100,
            'conflict_risk': ['Low']*100,
            'alertness_label': ['Calm']*100
        })
        df.to_csv('dummy_data.csv', index=False)
    
    dataset = BehaviourDataset('dummy_data.csv')
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    for x, y, r, a in loader:
        print(f"Batch shape: {x.shape}, Labels: {y}, Risks: {r}, Alerts: {a}")
        break
    
    # Cleanup
    if os.path.exists('dummy_data.csv'):
        os.remove('dummy_data.csv')
