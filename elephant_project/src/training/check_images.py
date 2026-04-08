import os
import cv2
from pathlib import Path

def check_images(root_dir):
    print(f"Checking images in {root_dir}...")
    bad_files = []
    count = 0
    for filepath in Path(root_dir).rglob('*'):
        if filepath.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            try:
                img = cv2.imread(str(filepath))
                if img is None:
                    print(f"Corrupt (None): {filepath}")
                    bad_files.append(str(filepath))
                else:
                    count += 1
            except Exception as e:
                print(f"Corrupt ({e}): {filepath}")
                bad_files.append(str(filepath))
                
    print(f"Checked {count} images.")
    with open('bad_images.txt', 'w') as f:
        if bad_files:
            print(f"Found {len(bad_files)} corrupt images:")
            for filename in bad_files:
                print(filename)
                f.write(filename + '\n')
        else:
            print("No corrupt images found.")

if __name__ == "__main__":
    check_images(r"c:\elephant posture and behaviour\elephant_project\elephant_dataset\posture_split")
