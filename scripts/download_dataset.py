import requests
from tqdm import tqdm
import os

def download_file_from_zenodo(url, output_path):
    """
    Scarica un file di grandi dimensioni da Zenodo (o un generico URL)
    mostrando una barra di avanzamento.
    """
    response = requests.get(url, stream=True)
    total_size_in_bytes= int(response.headers.get('content-length', 0))
    
    # Streaming chunk
    block_size = 1024
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    
    with open(output_path, 'wb') as f:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)
    progress_bar.close()
    
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERRORE: Download incompleto.")

if __name__ == "__main__":
    # Esempio di link: la pagina Zenodo in questione può rimandare a un file .zip/.tar
    # Verifica l'URL diretto del file .zip e sostituisci "DOWNLOAD_URL"
    DOWNLOAD_URL = "https://zenodo.org/record/5528418/files/openforensics_dataset_part1.zip"
    
    # Nome file in output
    OUTPUT_FILE = os.path.join("data", "openforensics_dataset_part1.zip")
    
    print("Inizio download del dataset da Zenodo...")
    download_file_from_zenodo(DOWNLOAD_URL, OUTPUT_FILE)
    print(f"Download completato, file salvato in {OUTPUT_FILE}")