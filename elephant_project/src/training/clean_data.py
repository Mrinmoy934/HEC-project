import os

def clean_data():
    if not os.path.exists('bad_images.txt'):
        print("bad_images.txt not found.")
        return

    with open('bad_images.txt', 'r') as f:
        lines = f.readlines()
    
    print(f"Found {len(lines)} bad files to delete.")
    for line in lines:
        filepath = line.strip()
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
                print(f"Deleted: {filepath}")
            except Exception as e:
                print(f"Error deleting {filepath}: {e}")
        else:
            print(f"File not found: {filepath}")

if __name__ == "__main__":
    clean_data()
