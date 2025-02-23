import requests
from tqdm import tqdm
import os

def download_file_from_zenodo(url, output_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()  # genererà un'eccezione se il download fallisce

    total_size_in_bytes = int(response.headers.get('content-length', 0))
    
    block_size = 1024
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True, desc=os.path.basename(output_path))
    
    with open(output_path, 'wb') as f:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)
    progress_bar.close()
    
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR: Incomplete download.")

if __name__ == "__main__":
    # Esempio di URL: singolo file, oppure ne gestisci diversi in un loop
    DOWNLOAD_URL = "https://zenodo.org/api/records/5528418/files-archive"

    # Ora salviamo in data/dataset
    os.makedirs("data/dataset", exist_ok=True)
    OUTPUT_FILE = os.path.join("data", "dataset", "openforensics_dataset_part1.zip")
    
    print("Starting dataset download from Zenodo...")
    download_file_from_zenodo(DOWNLOAD_URL, OUTPUT_FILE)
    print(f"Download completed, file saved in {OUTPUT_FILE}")
