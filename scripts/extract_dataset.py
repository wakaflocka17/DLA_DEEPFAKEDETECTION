import os
import zipfile
import argparse
import shutil

def extract_dataset(dataset_dir):
    """
    Reads all files in dataset_dir and, based on the filename, places them 
    into the corresponding folders:
      - Train_part_*.zip      -> extract into data/Train/Train
      - Train_poly.json       -> move to data/Train
      - Val_part_*.zip        -> extract into data/Val/Val
      - Val_poly.json         -> move to data/Val
      - Test-Dev_part_*.zip   -> extract into data/Test-Dev/Test-Dev
      - Test-Dev_poly.json    -> move to data/Test-Dev
      - Test-Challenge_part_*.zip  -> extract into data/Test-Challenge/Test-Challenge
      - Test-Challenge_poly.json   -> move to data/Test-Challenge
    """
    
    # Define destination folders
    base_data_dir = "data"
    train_dir = os.path.join(base_data_dir, "Train")
    val_dir = os.path.join(base_data_dir, "Val")
    test_dev_dir = os.path.join(base_data_dir, "Test-Dev")
    test_challenge_dir = os.path.join(base_data_dir, "Test-Challenge")

    # Create the necessary subfolders
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dev_dir, exist_ok=True)
    os.makedirs(test_challenge_dir, exist_ok=True)

    # Scan all files in dataset_dir
    for filename in os.listdir(dataset_dir):
        filepath = os.path.join(dataset_dir, filename)

        if os.path.isfile(filepath):
            # Check prefix and/or extension
            if filename.startswith("Train_part_") and filename.endswith(".zip"):
                # Example: "Train_part_1.zip"
                extract_zip(filepath, train_dir)
            
            elif filename.startswith("Train_poly") and filename.endswith(".json"):
                # Example: "Train_poly.json"
                shutil.copy(filepath, train_dir)

            elif filename.startswith("Val") and filename.endswith(".zip"):
                extract_zip(filepath, val_dir)

            elif filename.startswith("Val_poly") and filename.endswith(".json"):
                shutil.copy(filepath, val_dir)

            elif filename.startswith("Test-Dev_part_") and filename.endswith(".zip"):
                extract_zip(filepath, test_dev_dir)

            elif filename.startswith("Test-Dev_poly") and filename.endswith(".json"):
                shutil.copy(filepath, test_dev_dir)

            elif filename.startswith("Test-Challenge_part_") and filename.endswith(".zip"):
                extract_zip(filepath, test_challenge_dir)

            elif filename.startswith("Test-Challenge_poly") and filename.endswith(".json"):
                shutil.copy(filepath, test_challenge_dir)

            else:
                print(f"Skipping file (unrecognized): {filename}")

def extract_zip(zip_path, dest_dir):
    """
    Extracts the zip file into dest_dir
    """
    print(f"Extracting {os.path.basename(zip_path)} -> {dest_dir}")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(dest_dir)
    print("Extraction done.\n")

def main():
    parser = argparse.ArgumentParser(description="Extract and place the dataset partitions into the correct subfolders.")
    parser.add_argument("--dataset_dir", type=str, default="data/dataset",
                        help="Directory containing the downloaded ZIP/JSON files (default: data/dataset)")
    args = parser.parse_args()

    extract_dataset(args.dataset_dir)
    print("✅ All dataset files have been distributed correctly.")

if __name__ == "__main__":
    main()
