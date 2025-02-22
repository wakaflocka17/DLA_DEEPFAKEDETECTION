import os
import zipfile
import argparse

def extract_all_zips(input_dir, output_dir):
    """
    Extracts all .zip files in input_dir, saving their contents in output_dir.
    """
    # Check to ensure the destination folder exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".zip"):
            zip_path = os.path.join(input_dir, filename)
            print(f"Extracting: {zip_path} → {output_dir}")
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(output_dir)
            except zipfile.BadZipFile:
                print(f"Error: {filename} is corrupted or not a valid zip file.")
            else:
                print(f"✓ Extraction completed for: {filename}\n")

def main():
    parser = argparse.ArgumentParser(description="Extracts all .zip files from a folder")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Folder containing the zip files")
    parser.add_argument("--output_dir", type=str, default="data",
                        help="Destination folder (default: data)")
    args = parser.parse_args()

    extract_all_zips(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()
