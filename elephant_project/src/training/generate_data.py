import pandas as pd
import numpy as np

def create_dummy_data(filename='behaviour_dataset.csv'):
    num_frames = 1000
    df = pd.DataFrame({
        'frame_id': range(num_frames),
        'elephant_id': [1]*(num_frames//2) + [2]*(num_frames - num_frames//2),
        'posture_class': np.random.randint(0, 16, num_frames),
        'x1': np.random.randint(0, 500, num_frames),
        'y1': np.random.randint(0, 500, num_frames),
        'x2': np.random.randint(500, 1000, num_frames),
        'y2': np.random.randint(500, 1000, num_frames),
        'move_dx': np.random.rand(num_frames),
        'move_dy': np.random.rand(num_frames),
        'trunk_angle': np.random.randint(0, 360, num_frames),
        'ear_freq': np.random.rand(num_frames),
        'tail_freq': np.random.rand(num_frames),
        'behaviour_label': np.random.choice(['Calm', 'Aggressive', 'Alert'], num_frames),
        'conflict_risk': np.random.choice(['Low', 'Medium', 'High'], num_frames),
        'alertness_label': np.random.choice(['Calm', 'Aggressive'], num_frames)
    })
    df.to_csv(filename, index=False)
    print(f"Created {filename} with {num_frames} rows.")

if __name__ == "__main__":
    create_dummy_data()
