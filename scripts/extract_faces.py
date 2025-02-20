import os
import cv2
import json
import argparse

def extract_faces(annotations_path, images_root, output_root):
    """
    Legge un file di annotazioni contenente bounding box e label (real/fake),
    carica ogni immagine e ritaglia i volti.
    
    :param annotations_path: Path al file JSON (o CSV) che contiene info su bounding box e label
    :param images_root: Cartella dove si trovano le immagini originali
    :param output_root: Cartella dove salvare i volti ritagliati
    """
    
    # Crea le cartelle 'real' e 'fake' se non esistono
    real_dir = os.path.join(output_root, "real")
    fake_dir = os.path.join(output_root, "fake")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    # Leggi le annotazioni (formato di esempio: JSON con campi img_path, bboxes, label)
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
    
    # Esempio di struttura di annotations (JSON):
    # [
    #   {
    #       "img_name": "0001.jpg",
    #       "bboxes": [ {"x1": 10, "y1": 20, "x2": 110, "y2": 160, "label": "real"}, {...} ],
    #   },
    #   ...
    # ]
    
    face_count = 0
    
    for ann in annotations:
        img_name = ann["img_name"]
        bboxes = ann["bboxes"]

        # Percorso assoluto dell'immagine
        img_path = os.path.join(images_root, img_name)
        
        if not os.path.isfile(img_path):
            print(f"Attenzione: {img_path} non trovato!")
            continue
        
        image = cv2.imread(img_path)
        
        if image is None:
            print(f"Impossibile aprire {img_path}")
            continue
        
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
            label = bbox["label"]  # "real" o "fake"
            
            # Ritaglia il volto
            cropped_face = image[y1:y2, x1:x2]
            
            # Salvataggio del volto
            face_filename = f"{os.path.splitext(img_name)[0]}_{face_count}.jpg"
            
            if label == "real":
                save_path = os.path.join(real_dir, face_filename)
            else:
                save_path = os.path.join(fake_dir, face_filename)
            
            cv2.imwrite(save_path, cropped_face)
            face_count += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=str, required=True,
                        help="Path al file JSON con bounding box e label")
    parser.add_argument("--images_root", type=str, required=True,
                        help="Cartella con le immagini originali")
    parser.add_argument("--output_root", type=str, default="data/cropped_faces",
                        help="Cartella dove salvare i volti ritagliati")
    args = parser.parse_args()

    extract_faces(args.annotations, args.images_root, args.output_root)

if __name__ == "__main__":
    main()
