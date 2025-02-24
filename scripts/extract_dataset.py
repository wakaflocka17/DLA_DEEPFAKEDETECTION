import os
import zipfile
import argparse
import shutil

def extract_dataset(dataset_dir):
    """
    Legge tutti i file in dataset_dir e, in base al nome, li colloca
    nelle cartelle corrispondenti:
      - Train_part_*.zip      -> estrai in data/Train/Train
      - Train_poly.json       -> sposta in data/Train
      - Val_part_*.zip        -> estrai in data/Val/Val
      - Val_poly.json         -> sposta in data/Val
      - Test-Dev_part_*.zip   -> estrai in data/Test-Dev/Test-Dev
      - Test-Dev_poly.json    -> sposta in data/Test-Dev
      - Test-Challenge_part_*.zip  -> estrai in data/Test-Challenge/Test-Challenge
      - Test-Challenge_poly.json   -> sposta in data/Test-Challenge
    """
    
    # Definiamo le cartelle di destinazione
    base_data_dir = "data"
    train_dir = os.path.join(base_data_dir, "Train")
    val_dir = os.path.join(base_data_dir, "Val")
    test_dev_dir = os.path.join(base_data_dir, "Test-Dev")
    test_challenge_dir = os.path.join(base_data_dir, "Test-Challenge")
    
    # Creiamo eventuali sottocartelle mancanti
    os.makedirs(os.path.join(train_dir, "Train"), exist_ok=True)
    os.makedirs(os.path.join(val_dir, "Val"), exist_ok=True)
    os.makedirs(os.path.join(test_dev_dir, "Test-Dev"), exist_ok=True)
    os.makedirs(os.path.join(test_challenge_dir, "Test-Challenge"), exist_ok=True)
    
    # Scansiona tutti i file in dataset_dir
    for filename in os.listdir(dataset_dir):
        filepath = os.path.join(dataset_dir, filename)

        if os.path.isfile(filepath):
            # Controlliamo prefisso e/o estensione
            if filename.startswith("Train_part_") and filename.endswith(".zip"):
                # Esempio: "Train_part_1.zip"
                extract_zip(filepath, os.path.join(train_dir, "Train"))
            
            elif filename.startswith("Train_poly") and filename.endswith(".json"):
                # Esempio: "Train_poly.json"
                shutil.copy(filepath, train_dir)

            elif filename.startswith("Val") and filename.endswith(".zip"):
                extract_zip(filepath, os.path.join(val_dir, "Val"))

            elif filename.startswith("Val_poly") and filename.endswith(".json"):
                shutil.copy(filepath, val_dir)

            elif filename.startswith("Test-Dev_part_") and filename.endswith(".zip"):
                extract_zip(filepath, os.path.join(test_dev_dir, "Test-Dev"))

            elif filename.startswith("Test-Dev_poly") and filename.endswith(".json"):
                shutil.copy(filepath, test_dev_dir)

            elif filename.startswith("Test-Challenge_part_") and filename.endswith(".zip"):
                extract_zip(filepath, os.path.join(test_challenge_dir, "Test-Challenge"))

            elif filename.startswith("Test-Challenge_poly") and filename.endswith(".json"):
                shutil.copy(filepath, test_challenge_dir)

            else:
                print(f"Skipping file (unrecognized): {filename}")

def extract_zip(zip_path, dest_dir):
    """
    Estrae lo zip in dest_dir
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