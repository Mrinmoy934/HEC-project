import os
import shutil
import random
from pathlib import Path

def split_dataset(source_dir, output_dir, split_ratio=0.8):
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    if output_path.exists():
        shutil.rmtree(output_path)
    
    # Create train and val directories
    (output_path / 'train').mkdir(parents=True, exist_ok=True)
    (output_path / 'val').mkdir(parents=True, exist_ok=True)
    
    # Iterate over class directories
    for class_dir in source_path.iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            images = [f for f in class_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp', '.bmp']]
            
            random.shuffle(images)
            split_idx = int(len(images) * split_ratio)
            
            train_images = images[:split_idx]
            val_images = images[split_idx:]
            
            # Create class directories in output
            (output_path / 'train' / class_name).mkdir(exist_ok=True)
            (output_path / 'val' / class_name).mkdir(exist_ok=True)
            
            # Copy files
            for img in train_images:
                shutil.copy2(img, output_path / 'train' / class_name / img.name)
                
            for img in val_images:
                shutil.copy2(img, output_path / 'val' / class_name / img.name)
                
            print(f"Class '{class_name}': {len(train_images)} train, {len(val_images)} val")

if __name__ == "__main__":
    SOURCE_DIR = r"c:\elephant posture and behaviour\elephant_project\elephant_dataset\posture"
    OUTPUT_DIR = r"c:\elephant posture and behaviour\elephant_project\elephant_dataset\posture_split"
    
    print(f"Splitting dataset from {SOURCE_DIR} to {OUTPUT_DIR}...")
    split_dataset(SOURCE_DIR, OUTPUT_DIR)
    print("Dataset split complete.")
