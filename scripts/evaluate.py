import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
import csv
import os

from dataloader import create_dataloader
from deepfake_classifier import get_model

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else 
    "mps" if torch.backends.mps.is_available() else 
    "cpu"
)

def plot_accuracy_vs_resolution(model_name="custom", dataset="Test-Dev", resolutions=[64, 128, 256, 512]):
    """
    Valuta il modello deepfake per diverse risoluzioni delle immagini e traccia un grafico di Accuracy vs. Risoluzione.
    
    Args:
        model_name (str): Nome del modello ('mobilenet', 'xception' o 'custom').
        dataset (str): Nome del dataset di test, ad esempio 'Test-Dev' o 'Test-Challenge'.
        resolutions (list): Lista di risoluzioni (larghezza/altezza in pixel, assumendo immagini quadrate) da valutare.
    """
    accuracies = []

    # Per ogni risoluzione, creiamo un dataloader che ridimensiona le immagini a quella risoluzione
    for res in resolutions:
        print(f"\nValutazione a risoluzione {res}x{res}...")
        # Si assume che create_dataloader accetti il parametro "resolution" per applicare la trasformazione
        test_loader = create_dataloader(f"processed_data/{dataset}", batch_size=32, shuffle=False, resolution=res)
        
        # Carichiamo il modello e i relativi pesi
        model = get_model(model_name).to(DEVICE)
        model.load_state_dict(torch.load(f"models/{model_name}_deepfake.pth", map_location=DEVICE))
        model.eval()

        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calcoliamo l'accuratezza per questa risoluzione
        acc = accuracy_score(all_labels, all_preds)
        accuracies.append(acc)
        print(f"Risoluzione {res}x{res} - Accuracy: {acc:.4f}")

    # Tracciamo il grafico: Risoluzione in x e Accuracy in y
    plt.figure(figsize=(7, 5))
    plt.plot(resolutions, accuracies, marker='o')
    plt.title(f"Accuracy vs. Risoluzione Immagine ({model_name})")
    plt.xlabel("Risoluzione Immagine (px)")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_all_models(models=["mobilenet","xception","custom"]):
    """
    Genera due grafici separati per confrontare:
    - Training Loss: curv di tutti i modelli
    - Validation Accuracy: curve di tutti i modelli

    Legge i rispettivi CSV salvati durante il training (es. logs/mobilenet_train_logs.csv)
    e traccia tre linee per ciascun grafico (una per modello).
    """

    # Dizionario in cui memorizziamo i dati caricati: data[model_name] = { 'epochs': [...], 'loss': [...], 'acc': [...] }
    data = {}

    # 1) Carichiamo i file CSV per ogni modello
    for model_name in models:
        csv_path = f"logs/{model_name}_train_logs.csv"
        if not os.path.exists(csv_path):
            print(f"Log file {csv_path} not found. Skipping '{model_name}'.")
            continue

        epochs = []
        train_losses = []
        val_accs = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                epochs.append(int(row["epoch"]))
                train_losses.append(float(row["train_loss"]))
                val_accs.append(float(row["val_accuracy"]))

        data[model_name] = {
            "epochs": epochs,
            "loss": train_losses,
            "acc": val_accs
        }

    # Se non è stato trovato alcun file, interrompiamo
    if not data:
        print("No data to plot. Make sure the CSV logs exist.")
        return

    # 2) PRIMO GRAFICO: Training Loss di tutti i modelli
    plt.figure(figsize=(7,5))
    for model_name, model_data in data.items():
        plt.plot(model_data["epochs"], model_data["loss"], label=model_name)
    plt.title("Training Loss Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()  # oppure plt.savefig("loss_comparison.png"); plt.close()

    # 3) SECONDO GRAFICO: Validation Accuracy di tutti i modelli
    plt.figure(figsize=(7,5))

    all_acc_values = []
    for model_name, model_data in data.items():
        plt.plot(model_data["epochs"], model_data["acc"], label=model_name)
        all_acc_values.extend(model_data["acc"])

    plt.title("Validation Accuracy Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    # Calcoliamo min e max di TUTTE le accuracy
    min_acc = min(all_acc_values)
    max_acc = max(all_acc_values)

    # Impostiamo un piccolo margine per non tagliare i punti estremi
    margin = 0.01
    lower_ylim = max(0, min_acc - margin)
    upper_ylim = min(1, max_acc + margin)

    plt.ylim([lower_ylim, upper_ylim])  # Imposta i limiti dinamici

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def evaluate(model_name="mobilenet", dataset="Test-Dev"):
    """
    Valuta il modello deepfake su un dataset di test.
    :param model_name: 'mobilenet', 'xception' o 'custom'
    :param dataset: 'Test-Dev' o 'Test-Challenge'
    """

    # 1) Carico il dataloader di test
    test_loader = create_dataloader(f"processed_data/{dataset}", batch_size=32, shuffle=False)

    # 2) Carico il modello allenato
    model = get_model(model_name).to(DEVICE)
    model.load_state_dict(torch.load(f"models/{model_name}_deepfake.pth", map_location=DEVICE))
    model.eval()

    all_preds = []
    all_labels = []

    print(f"\n🔍 Evaluating {model_name} on {dataset}...\n")

    # 3) Calcolo delle predizioni su tutto il test set
    progress_bar = tqdm(test_loader, desc="Evaluating", leave=True)
    with torch.no_grad():
        for images, labels in progress_bar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 4) Calcolo delle metriche
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print(f"\n📊 Evaluation Results ({dataset}, {model_name}):")
    print(f"✔️ Accuracy:  {acc:.4f}")
    print(f"✔️ Precision: {prec:.4f}")
    print(f"✔️ Recall:    {rec:.4f}")
    print(f"✔️ F1-score:  {f1:.4f}\n")

    # 5) Mostro il grafico di Training Loss e Validation Accuracy 
    #    (caricando i dati dal CSV salvato durante il training)
    plot_training_curves(model_name)


def plot_training_curves(model_name):
    """
    Legge i log di training (loss e accuracy) dal CSV e mostra un grafico
    con Training Loss e Validation Accuracy in funzione dell'epoca per il singolo modello.
    """
    csv_file = f"logs/{model_name}_train_logs.csv"
    if not os.path.exists(csv_file):
        print(f"❌ Log file {csv_file} not found. Unable to plot training curves.")
        return

    epochs = []
    train_losses = []
    val_accuracies = []

    # Leggo i dati dal CSV
    with open(csv_file, mode="r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_losses.append(float(row["train_loss"]))
            val_accuracies.append(float(row["val_accuracy"]))

    # Creiamo una figura con due subplot affiancati
    plt.figure(figsize=(12,5))

    # Subplot 1: Training Loss
    plt.subplot(1,2,1)
    plt.plot(epochs, train_losses, label='Training Loss', color='red')
    plt.title(f'{model_name} - Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Subplot 2: Validation Accuracy
    plt.subplot(1,2,2)
    plt.plot(epochs, val_accuracies, label='Validation Accuracy', color='blue')
    plt.title(f'{model_name} - Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    # Calcolo dei limiti dinamici per l'asse y
    min_acc = min(val_accuracies)
    max_acc = max(val_accuracies)
    margin = 0.01
    lower_ylim = max(0, min_acc - margin)
    upper_ylim = min(1, max_acc + margin)
    plt.ylim([lower_ylim, upper_ylim])
    
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Se preferisci salvare il grafico invece che mostrarlo a schermo:
    # plt.savefig(f"{model_name}_training_plot.png")
    # plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate deepfake detection model")
    parser.add_argument("--model", type=str, choices=["mobilenet", "xception", "custom"], required=True,
                        help="Scegli il modello: 'mobilenet', 'xception' o 'custom'")
    parser.add_argument("--dataset", type=str, choices=["Test-Dev", "Test-Challenge"], required=True,
                        help="Dataset: 'Test-Dev' o 'Test-Challenge'")
    parser.add_argument("--plot_all", action="store_true",
                        help="Se presente, genera un grafico di confronto globale per tutti i modelli")
    args = parser.parse_args()

    # Esegui la valutazione per il singolo modello richiesto
    evaluate(args.model, args.dataset)

    # Se l'utente richiede il confronto globale, chiama plot_all_models
    if args.plot_all:
        plot_all_models(["mobilenet", "xception", "custom"])

    # Esegui la valutazione dell'Accuracy al variare della risoluzione dell'immagine
    plot_accuracy_vs_resolution(model_name="custom", dataset="Test-Dev", resolutions=[64, 128, 256, 512])