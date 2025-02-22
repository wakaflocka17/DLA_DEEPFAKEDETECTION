import os
import zipfile
import argparse

def extract_all_zips(input_dir, output_dir):
    """
    Estrae tutti i file .zip in input_dir, salvandone il contenuto in output_dir.
    """
    # Controllo per verificare che la cartella di destinazione esista
    os.makedirs(output_dir, exist_ok=True)

    # Iterazione su tutti i file nella cartella di input
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".zip"):
            zip_path = os.path.join(input_dir, filename)
            print(f"Estrazione di: {zip_path} → {output_dir}")
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(output_dir)
            except zipfile.BadZipFile:
                print(f"Errore: {filename} è corrotto o non è un file zip valido.")
            else:
                print(f"✓ Estrazione completata per: {filename}\n")

def main():
    parser = argparse.ArgumentParser(description="Estrae tutti i file .zip da una cartella")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Cartella che contiene i file zip")
    parser.add_argument("--output_dir", type=str, default="data",
                        help="Cartella di destinazione (default: data)")
    args = parser.parse_args()

    extract_all_zips(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()