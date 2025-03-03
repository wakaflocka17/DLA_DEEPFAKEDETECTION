import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, average_precision_score
import seaborn as sns

# Function to calculate Intersection over Union (IoU)
def iou(boxA, boxB):
    """ Compute Intersection over Union (IoU) between two bounding boxes. """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    denominator = float(boxAArea + boxBArea - interArea)
    if denominator == 0:  # Avoid division by zero
        return 0.0

    return interArea / denominator

# Function to calculate Localization Recall Precision (LRP) Error
def compute_lrp(ground_truth, predictions):
    precision = precision_score(ground_truth, predictions)
    recall = recall_score(ground_truth, predictions)
    lrp_error = 1 - ((2 * precision * recall) / (precision + recall))
    return lrp_error

# Function to calculate Average Precision (AP)
def compute_ap(ground_truth, scores):
    ap = average_precision_score(ground_truth, scores)
    return ap

# Function to evaluate face extraction (separately for "real" and "fake")
def evaluate_face_extraction(json_path, extracted_faces_path):
    """ Compare extracted face bounding boxes with ground truth, considering 'real' and 'fake' separately. """
    
    # Load original bounding boxes from JSON
    with open(json_path, "r") as f:
        data = json.load(f)

    results = []
    
    for category_name, category_id in [("real", 0), ("fake", 1)]:
        total_iou = []
        ground_truth_labels = []
        predicted_labels = []
        prediction_scores = []

        for annotation in data["annotations"]:
            if annotation["category_id"] != category_id:
                continue  # Skip annotations that do not belong to the correct category
            
            bbox = annotation["bbox"]  # [x, y, width, height]
            bbox_gt = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]  # Convert to [x1, y1, x2, y2]

            # Simulate an extracted bounding box (MUST BE REPLACED WITH YOUR LOGIC!)
            bbox_extracted = [bbox[0] + 5, bbox[1] + 5, bbox[0] + bbox[2] - 5, bbox[1] + bbox[3] - 5]

            # Compute IoU between ground truth and extracted bounding box
            iou_value = iou(bbox_gt, bbox_extracted)
            total_iou.append(iou_value)

            # If IoU > 0.5, consider the bounding box as correctly extracted
            ground_truth_labels.append(1)
            predicted_labels.append(1 if iou_value > 0.5 else 0)
            prediction_scores.append(iou_value)  # Use IoU as probability for AP

        # Compute metrics
        mean_iou = np.mean(total_iou) if total_iou else 0
        lrp_error = compute_lrp(ground_truth_labels, predicted_labels) if ground_truth_labels else 1.0
        ap = compute_ap(ground_truth_labels, prediction_scores) if ground_truth_labels else 0

        results.append([category_name, mean_iou, lrp_error, ap])

    return results

# Dataset splits and corresponding folders
splits = {
    "Train": ("data/Train/Train_poly.json", "processed_data/train_cropped"),
    "Val": ("data/Val/Val_poly.json", "processed_data/val_cropped"),
    "Test-Dev": ("data/Test-Dev/Test-Dev_poly.json", "processed_data/test_dev_cropped"),
    "Test-Challenge": ("data/Test-Challenge/Test-Challenge_poly.json", "processed_data/test_challenge_cropped")
}

# Compute metrics for each dataset and category (real/fake)
all_results = []
for split, (json_path, extracted_faces_path) in splits.items():
    print(f"🔍 Evaluating {split} dataset...")
    results = evaluate_face_extraction(json_path, extracted_faces_path)
    for category, mean_iou, lrp_error, ap in results:
        all_results.append([split, category, mean_iou, lrp_error, ap])

# Create DataFrame to visualize results
df_results = pd.DataFrame(all_results, columns=["Dataset", "Category", "Mean IoU", "LRP Error", "AP Score"])
print("\n📊 Evaluation Results:\n")
print(df_results)

# Display results table with Seaborn
plt.figure(figsize=(10, 5))
sns.heatmap(df_results.pivot(index="Dataset", columns="Category", values="Mean IoU"), 
            annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Face Extraction Evaluation - IoU")
plt.show()

plt.figure(figsize=(10, 5))
sns.heatmap(df_results.pivot(index="Dataset", columns="Category", values="LRP Error"), 
            annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Face Extraction Evaluation - LRP Error")
plt.show()

plt.figure(figsize=(10, 5))
sns.heatmap(df_results.pivot(index="Dataset", columns="Category", values="AP Score"), 
            annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Face Extraction Evaluation - AP Score")
plt.show()
