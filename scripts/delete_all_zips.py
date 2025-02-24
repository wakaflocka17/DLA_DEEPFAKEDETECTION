import os
import glob

def delete_all_zips(directory="data"):
    """
    Deletes all .zip files in the specified directory.

    :param directory: Path to the directory where .zip files will be deleted (default: "data/")
    """
    zip_files = glob.glob(os.path.join(directory, "*.zip"))  # Trova tutti i file .zip in "data/"

    if not zip_files:
        print(f"✅ No .zip files found in {directory}")
        return
    
    for zip_file in zip_files:
        try:
            os.remove(zip_file)
            print(f"🗑️ Deleted: {zip_file}")
        except Exception as e:
            print(f"❌ Error deleting {zip_file}: {e}")

    print(f"\n✅ All .zip files in '{directory}' have been deleted.")

if __name__ == "__main__":
    delete_all_zips()