import torch
import torch.nn as nn
from tqdm import tqdm  # Barra di avanzamento
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

from scripts.dataloader import create_dataloader
from deepfake_classifier import get_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def evaluate(model_name="mobilenet", dataset="Test-Dev"):
    """
    Evaluates the trained deepfake detection model.
    :param model_name: 'mobilenet' or 'xception'
    :param dataset: 'Test-Dev' or 'Test-Challenge'
    """

    # Load dataset
    test_loader = create_dataloader(f"processed_data/{dataset}", batch_size=32, shuffle=False)

    # Load model
    model = get_model(model_name).to(DEVICE)
    model.load_state_dict(torch.load(f"models/{model_name}_deepfake.pth", map_location=DEVICE))
    model.eval()

    all_preds = []
    all_labels = []

    print(f"\n🔍 Evaluating {model_name} on {dataset}...\n")

    # Barra di avanzamento
    progress_bar = tqdm(test_loader, desc="Evaluating", leave=True)

    with torch.no_grad():
        for images, labels in progress_bar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print(f"\n📊 Evaluation Results ({dataset}, {model_name}):")
    print(f"✔️ Accuracy:  {acc:.4f}")
    print(f"✔️ Precision: {prec:.4f}")
    print(f"✔️ Recall:    {rec:.4f}")
    print(f"✔️ F1-score:  {f1:.4f}\n")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate deepfake detection model")
    parser.add_argument("--model", type=str, choices=["mobilenet", "xception"], required=True, help="Choose model: 'mobilenet' or 'xception'")
    parser.add_argument("--dataset", type=str, choices=["Test-Dev", "Test-Challenge"], required=True, help="Dataset: 'Test-Dev' or 'Test-Challenge'")
    args = parser.parse_args()

    evaluate(args.model, args.dataset)
