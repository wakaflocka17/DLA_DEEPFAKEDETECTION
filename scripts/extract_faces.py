import os
import cv2
import json
import argparse
from tqdm import tqdm  # for the progress bar

def extract_faces(json_path, images_root, image_subdir, output_root):
    """
    Extracts Real/Fake faces from a COCO-style JSON file, with a dynamic image directory.
    """

    # Create 'real' and 'fake' folders
    real_dir = os.path.join(output_root, "real")
    fake_dir = os.path.join(output_root, "fake")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    # Read the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Map image_id -> file_name
    images_info = {img["id"]: img["file_name"] for img in data["images"]}

    face_count = 0

    for ann in tqdm(data["annotations"], desc=f"Processing annotations in {os.path.basename(json_path)}"):
        image_id = ann["image_id"]
        category_id = ann["category_id"]
        x, y, w, h = ann["bbox"]

        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)

        if image_id not in images_info:
            continue

        file_name = os.path.basename(images_info[image_id])  # Extract only the file name

        # Dynamic path based on image_subdir
        img_path = os.path.join(images_root, image_subdir, file_name) if image_subdir else os.path.join(images_root, file_name)

        if not os.path.isfile(img_path):
            print(f"Warning: {img_path} not found!")
            continue

        image = cv2.imread(img_path)
        if image is None:
            print(f"Unable to open {img_path}")
            continue

        cropped_face = image[y1:y2, x1:x2]

        label_str = "real" if category_id == 0 else "fake"
        face_filename = f"{os.path.splitext(file_name)[0]}_{face_count}.jpg"

        save_path = os.path.join(real_dir if category_id == 0 else fake_dir, face_filename)
        cv2.imwrite(save_path, cropped_face)
        face_count += 1

def main():
    datasets = [
        {"name": "Train", "json_path": "data/Train/Train_poly.json", "image_subdir": "Train/Train", "output_root": "processed_data/train_cropped"},
        {"name": "Validation", "json_path": "data/Val/Val_poly.json", "image_subdir": "Val/Val", "output_root": "processed_data/val_cropped"},
        {"name": "Test-Dev", "json_path": "data/Test-Dev/Test-Dev_poly.json", "image_subdir": "Test-Dev/Test-Dev", "output_root": "processed_data/test_dev_cropped"},
        {"name": "Test-Challenge", "json_path": "data/Test-Challenge/Test-Challenge_poly.json", "image_subdir": "Test-Challenge/Test-Challenge", "output_root": "processed_data/test_challenge_cropped"}
    ]

    images_root = "data"  # Common base folder for all datasets

    for dataset in datasets:
        print(f"\n🔄 Processing dataset: {dataset['name']}...")
        extract_faces(dataset["json_path"], images_root, dataset["image_subdir"], dataset["output_root"])
        print(f"✅ Completed: {dataset['name']} → Faces extracted in {dataset['output_root']}")

if __name__ == "__main__":
    main()
