import requests
from tqdm import tqdm
import os

def download_file_from_zenodo(url, output_path):
    """
    Downloads the dataset hosted on Zenodo
    """
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    
    # Streaming chunk
    block_size = 1024
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    
    with open(output_path, 'wb') as f:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)
    progress_bar.close()
    
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR: Incomplete download.")

if __name__ == "__main__":
    # The Zenodo page in question links to a .zip/.tar file
    DOWNLOAD_URL = "https://zenodo.org/api/records/5528418/files-archive"
    
    # Output file name
    OUTPUT_FILE = os.path.join("data", "openforensics_dataset_part1.zip")
    
    print("Starting dataset download from Zenodo...")
    download_file_from_zenodo(DOWNLOAD_URL, OUTPUT_FILE)
    print(f"Download completed, file saved in {OUTPUT_FILE}")
