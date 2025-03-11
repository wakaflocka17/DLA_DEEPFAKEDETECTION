import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm  # Barra di avanzamento
import os
import csv

from dataloader import create_dataloader
from deepfake_classifier import get_model

# Training parameters
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4

# Se hai una GPU Nvidia, userà CUDA, altrimenti proverà MPS su Apple Silicon, altrimenti CPU.
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else 
    "mps" if torch.backends.mps.is_available() else 
    "cpu"
)

def train(model_name="mobilenet"):
    """
    Trains the deepfake classifier.
    :param model_name: 'mobilenet', 'xception' o 'custom'
    """

    # 1) Carico i dataloader per train e validation
    train_loader = create_dataloader("processed_data/train_cropped", batch_size=BATCH_SIZE)
    val_loader   = create_dataloader("processed_data/val_cropped",   batch_size=BATCH_SIZE, shuffle=False)

    # 2) Inizializzo il modello, la loss e l'optimizer
    model = get_model(model_name).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\n🚀 Training {model_name} model on {DEVICE}...\n")

    # 3) Preparo un file CSV per salvare i log di training e validazione
    os.makedirs("logs", exist_ok=True)
    csv_file = f"logs/{model_name}_train_logs.csv"
    with open(csv_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_accuracy"])  # header

    # 4) Loop sulle epoche
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        # Barra di avanzamento per il training
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=True, position=0)
        for images, labels in progress_bar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        # Calcolo la loss media su tutto il train set
        avg_loss = running_loss / len(train_loader)

        # 5) Calcolo l'accuracy sul validation set
        val_acc = evaluate_on_validation(model, val_loader)

        # Stampo i risultati di questa epoca
        print(f"✅ Epoch {epoch+1}/{EPOCHS} - Training Loss: {avg_loss:.4f} - Val Accuracy: {val_acc:.4f}")

        # 6) Salvo i risultati di questa epoca nel file CSV
        with open(csv_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, avg_loss, val_acc])

    # 7) Salvo il modello allenato
    model_save_path = f"models/{model_name}_deepfake.pth"
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"📁 Model saved at {model_save_path}\n")


def evaluate_on_validation(model, val_loader):
    """
    Funzione di appoggio per calcolare l'accuracy sul validation set.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    model.train()  # Torno in modalità train
    return correct / total


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train deepfake detection model")
    parser.add_argument("--model", type=str, choices=["mobilenet", "xception", "custom"], required=True,
                        help="Scegli il modello: 'mobilenet', 'xception' o 'custom'")
    args = parser.parse_args()

    train(args.model)