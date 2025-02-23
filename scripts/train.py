import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm  # Barra di avanzamento
import os

from dataloader import create_dataloader
from deepfake_classifier import get_model

# Training parameters
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def train(model_name="mobilenet"):
    """
    Trains the deepfake classifier.
    :param model_name: 'mobilenet' or 'xception'
    """

    # Load dataset (corrected paths!)
    train_loader = create_dataloader("processed_data/train_cropped", batch_size=BATCH_SIZE)
    val_loader = create_dataloader("processed_data/val_cropped", batch_size=BATCH_SIZE, shuffle=False)

    # Load model
    model = get_model(model_name).to(DEVICE)
    
    # Loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\n🚀 Training {model_name} model on {DEVICE}...\n")

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

        avg_loss = running_loss / len(train_loader)
        print(f"✅ Epoch {epoch+1}/{EPOCHS} completed - Loss: {avg_loss:.4f}")

    # Save model
    model_save_path = f"models/{model_name}_deepfake.pth"
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"📁 Model saved at {model_save_path}\n")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train deepfake detection model")
    parser.add_argument("--model", type=str, choices=["mobilenet", "xception"], required=True, help="Choose model: 'mobilenet' or 'xception'")
    args = parser.parse_args()

    train(args.model)