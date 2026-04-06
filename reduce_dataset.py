import os
import random
from tqdm import tqdm

# --- CONFIG ---
# Ensure this matches the path where your folders (0_closed_eye, etc.) are
BASE_DIR = os.path.expanduser("~/Documents/projects/img-proc/ddds/data_set")
TARGETS = {
    "train": 6000,
    "val": 1000
}

def reduce_data():
    for split, limit in TARGETS.items():
        split_path = os.path.join(BASE_DIR, split)
        if not os.path.exists(split_path):
            print(f"Skipping {split}: Folder not found.")
            continue

        # Get subfolders like 0_closed_eye, 1_open_eye, etc.
        categories = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]

        for cat in categories:
            cat_path = os.path.join(split_path, cat)
            # List all image files
            images = [f for f in os.listdir(cat_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            current_count = len(images)
            if current_count <= limit:
                print(f"Nothing to do for {split}/{cat}: Already has {current_count} images.")
                continue

            # Calculate how many to delete
            to_delete_count = current_count - limit
            to_delete_list = random.sample(images, to_delete_count)

            print(f"Reducing {split}/{cat}: {current_count} -> {limit}")
            for img_name in tqdm(to_delete_list, desc=f"Deleting from {cat}"):
                os.remove(os.path.join(cat_path, img_name))

if __name__ == "__main__":
    confirm = input(f"This will permanently DELETE extra images in {BASE_DIR}. Type 'yes' to proceed: ")
    if confirm.lower() == 'yes':
        reduce_data()
        print("\nDataset successfully reduced!")
    else:
        print("Operation cancelled.")
