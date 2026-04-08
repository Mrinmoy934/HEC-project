from ultralytics import YOLO
import os

def train_yolo(data_yaml_path, epochs=100, imgsz=640, model_size='m'):
    """
    Trains a YOLOv8 model for elephant posture detection.
    
    Args:
        data_yaml_path (str): Path to the data.yaml file.
        epochs (int): Number of training epochs.
        imgsz (int): Image size.
        model_size (str): Model size ('n', 's', 'm', 'l', 'x').
    """
    # Load a model
    model_name = f'yolov8{model_size}.pt'
    model = YOLO(model_name)  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        project='../models/yolo',
        name='elephant_posture_v8',
        exist_ok=True
    )
    
    print(f"Training completed. Best model saved at {results.save_dir}")

if __name__ == '__main__':
    # Example usage
    # Ensure data.yaml exists at this path before running
    DATA_YAML = os.path.abspath('../../dataset/data.yaml') 
    
    # Check if data.yaml exists, if not, create a dummy one for demonstration if needed, 
    # but in real scenario user provides it.
    if not os.path.exists(DATA_YAML):
        print(f"Warning: {DATA_YAML} not found. Please ensure dataset is prepared.")
    else:
        train_yolo(DATA_YAML)
