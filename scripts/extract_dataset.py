import os
import zipfile
import argparse
from pathlib import Path

def extract_dataset(zip_path, output_dir, check_dir):
    """
    Extracts the dataset zip file into the specified folder
    and checks for the presence of `real` and `fake` in another folder.

    Args:
        zip_path (str): path to the zip file
        output_dir (str): destination folder for extraction
        check_dir (str): path where the real/fake folders should be found
    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Extracting dataset from {zip_path} to {output_dir}...")
    
    try:
        # Extract the zip file into output_dir
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        print("Extraction completed successfully!")
        
        # If you want to check the presence of real/fake in another folder
        data_dir = Path(check_dir)
        real_dir = data_dir / "real"
        fake_dir = data_dir / "fake"
        
        if not (real_dir.exists() and fake_dir.exists()):
            print("\nWARNING: The 'real' and/or 'fake' folders were not found.")
            print("Make sure the zip file contains the correct structure:")
            print(f"  {check_dir}/")
            print("    ├── real/")
            print("    └── fake/")
            
    except zipfile.BadZipFile:
        print("Error: The zip file appears to be corrupted or invalid.")
    except Exception as e:
        print(f"Error during extraction: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Extracts the dataset zip file and checks the real/fake folder structure")
    parser.add_argument("--zip_path", type=str, required=True,
                        help="Path to the dataset zip file")
    parser.add_argument("--output_dir", type=str, default="data",
                        help="Destination folder for extraction (default: data)")
    parser.add_argument("--check_dir", type=str, default="processed_data",
                        help="Folder where to verify the presence of real/ and fake/ (default: processed_data)")
    
    args = parser.parse_args()
    extract_dataset(args.zip_path, args.output_dir, args.check_dir)

if __name__ == "__main__":
    main()
