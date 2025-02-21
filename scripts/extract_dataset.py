import os
import zipfile
import argparse
from pathlib import Path

def extract_dataset(zip_path, output_dir):
    """
    Estrae il file zip del dataset nella cartella specificata.
    
    Args:
        zip_path: percorso del file zip
        output_dir: cartella di destinazione
    """
    # Crea la cartella di output se non esiste
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Estrazione del dataset da {zip_path} a {output_dir}...")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        print("Estrazione completata con successo!")
        
        # Verifica la presenza delle cartelle real/fake
        data_dir = Path(output_dir)
        real_dir = data_dir / "real"
        fake_dir = data_dir / "fake"
        
        if not (real_dir.exists() and fake_dir.exists()):
            print("\nATTENZIONE: Le cartelle 'real' e/o 'fake' non sono state trovate.")
            print("Assicurati che il file zip contenga la struttura corretta:")
            print("  data/")
            print("    ├── real/")
            print("    └── fake/")
            
    except zipfile.BadZipFile:
        print("Errore: Il file zip sembra essere corrotto o non valido.")
    except Exception as e:
        print(f"Errore durante l'estrazione: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Estrae il dataset zip nella cartella data")
    parser.add_argument("--zip_path", type=str, required=True,
                      help="Percorso del file zip del dataset")
    parser.add_argument("--output_dir", type=str, default="data",
                      help="Cartella di destinazione (default: data)")
    
    args = parser.parse_args()
    extract_dataset(args.zip_path, args.output_dir)

if __name__ == "__main__":
    main() 