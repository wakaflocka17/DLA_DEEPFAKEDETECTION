import cv2
import numpy as np
import torch
from cnn_custom import CustomCNN
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import os

# Versione 1: Grad-CAM con due immagini (originale + Grad-CAM)
def generate_gradcam(model, image_path, target_layer, output_path):
    """
    Esegue Grad-CAM su una singola immagine e salva il risultato
    side-by-side con l'immagine originale.
    
    :param model: Modello PyTorch (CustomCNN) già in modalità eval().
    :param image_path: Percorso dell'immagine di input.
    :param target_layer: Ultimo layer convoluzionale da usare per Grad-CAM (es. model.conv3).
    :param output_path: File di output per salvare l'immagine risultante.
    """
    # Leggi l'immagine
    bgr_img = cv2.imread(image_path)
    if bgr_img is None:
        raise FileNotFoundError(f"Impossibile leggere il file: {image_path}")
    
    # Converti BGR -> RGB per la fase di Grad-CAM
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    # Ridimensiona a 224x224 (o la dimensione che usa il tuo modello)
    rgb_img_resized = cv2.resize(rgb_img, (224, 224), interpolation=cv2.INTER_AREA)

    # Normalizza in [0,1]
    rgb_img_norm = np.float32(rgb_img_resized) / 255.0

    # Preprocessa l'immagine per il modello
    input_tensor = preprocess_image(
        rgb_img_norm,
        mean=[0.485, 0.456, 0.406],  # se i tuoi mean/std sono diversi, sostituisci
        std=[0.229, 0.224, 0.225]
    )

    # Crea l'istanza GradCAM
    cam = GradCAM(model=model, target_layers=[target_layer])

    # Calcola la mappa di attivazione
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)  # None = classe più probabile
    grayscale_cam = grayscale_cam[0, :]  # Prima immagine del batch

    # Sovrapponi la mappa all'immagine ridimensionata
    cam_image = show_cam_on_image(rgb_img_norm, grayscale_cam, use_rgb=True)
    # show_cam_on_image restituisce un'immagine in formato RGB (uint8)

    # Ora creiamo una visualizzazione side-by-side:
    # - A sinistra l'immagine originale ridimensionata
    # - A destra la Grad-CAM
    original_bgr = cv2.cvtColor(rgb_img_resized, cv2.COLOR_RGB2BGR)   # torniamo a BGR
    cam_image_bgr = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)        # convertiamo la cam in BGR

    # Concatenazione orizzontale
    side_by_side = np.hstack((original_bgr, cam_image_bgr))

    # Assicuriamoci che la cartella di destinazione esista
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Salva l'immagine finale
    cv2.imwrite(output_path, side_by_side)
    print(f"Salvato side-by-side (originale + Grad-CAM) in {output_path}")

def main():
    # Carica il modello e i pesi addestrati
    model = CustomCNN()
    model.load_state_dict(torch.load('models/custom_deepfake.pth', map_location='cpu'))
    model.eval()
    
    # Seleziona il layer target (ad esempio, l'ultimo blocco conv)
    target_layer = model.conv3

    # Esempio con due immagini: "fake" e "real"
    images = {
        "fake": "data/Train/Train/d5e38b17d0.jpg",
        "real": "data/Train/Train/4355a9a2f2.jpg"
    }

    # Per ognuna, genera Grad-CAM e salva
    for label, img_path in images.items():
        output_filename = f"utils/images/{label}_gradcam_side_by_side.png"
        generate_gradcam(model, img_path, target_layer, output_filename)

if __name__ == '__main__':
    main()
