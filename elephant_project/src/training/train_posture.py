from ultralytics import YOLO
import os

def train_posture_model(data_dir, epochs=50, img_size=640):
    # Load a model
    model = YOLO('yolov8m-cls.pt')  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(data=data_dir, epochs=epochs, imgsz=img_size, project='runs/classify', name='elephant_posture')
    
    print("Training complete.")
    print(f"Results saved to {results.save_dir}")

if __name__ == "__main__":
    DATA_DIR = r"c:\elephant posture and behaviour\elephant_project\elephant_dataset\posture_split"
    
    if not os.path.exists(DATA_DIR):
        print(f"Data directory {DATA_DIR} not found. Please run prepare_posture_data.py first.")
    else:
        train_posture_model(DATA_DIR)
