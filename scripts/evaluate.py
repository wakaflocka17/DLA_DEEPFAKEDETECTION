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

def plot_all_models(models=["mobilenet","xception","custom"]):
    """
    Genera due grafici separati per confrontare:
    - Training Loss: curv di tutti i modelli
    - Validation Accuracy: curve di tutti i modelli

    Legge i rispettivi CSV salvati durante il training (es. logs/mobilenet_train_logs.csv)
    e traccia tre linee per ciascun grafico (una per modello).
    """
    import csv
    import os
    import matplotlib.pyplot as plt

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
    for model_name, model_data in data.items():
        plt.plot(model_data["epochs"], model_data["acc"], label=model_name)
    plt.title("Validation Accuracy Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim([0,1])  # se vuoi forzare il range 0..1
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()  # oppure plt.savefig("accuracy_comparison.png"); plt.close()

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
    con Training Loss e Validation Accuracy in funzione dell'epoca.
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

    # Plot con matplotlib
    plt.figure(figsize=(12,5))

    # Subplot 1: Training Loss
    plt.subplot(1,2,1)
    plt.plot(epochs, train_losses, label='Training Loss', color='red')
    plt.title(f'{model_name} - Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Subplot 2: Validation Accuracy
    plt.figure(figsize=(12,5))

    all_acc_values = []  # conterrà tutte le accuracy di tutti i modelli
    for model_name, model_data in data.items():
        plt.plot(model_data["epochs"], model_data["acc"], label=model_name)
        all_acc_values.extend(model_data["acc"])

    plt.title("Validation Accuracy Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    # Calcoliamo il min e max di TUTTE le accuracy
    min_acc = min(all_acc_values)
    max_acc = max(all_acc_values)

    # Definiamo un margine di “buffer” per non tagliare i punti estremi
    margin = 0.01
    lower_ylim = max(0, min_acc - margin)
    upper_ylim = min(1, max_acc + margin)

    # Impostiamo dinamicamente i limiti dell’asse y
    plt.ylim([lower_ylim, upper_ylim])

    plt.legend()
    plt.grid(True)
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