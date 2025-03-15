# 🎯 4. FINE TUNING OF MOBILENET AND XCEPTION

## 4.1 General Description of Fine-Tuning
**Fine-tuning** is a transfer learning technique used to adapt a pre-trained neural network — originally trained on large generic datasets — to a specific, specialized task. 
Instead of training a model from scratch, we adjust the weights for our DeepFake Detection task.

By using this approach we reduce training time and the net guarentees better performances with a smaller set of training data.

---

## 4.2 Models Chosen for Fine-Tuning

We selected two state-of-the-art pre-trained CNN models for fine-tuning:

- **MobileNet-v2** (via PyTorch torchvision library)
- **Xception** (via Timm library)

Both models are used for extracting high-level visual features, which we then adapt for our binary classification task (Real vs. Fake images).

---

## 4.3 Fine-Tuning Implementation Details
Here we describe the precise process of fine-tuning performed for each model.

### 4.3.1 MobileNet-v2

For MobileNet-v2, we replace the original classification layer to adapt the network to our binary classification task.

```python
model = models.mobilenet_v2(pretrained=True)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 2)
```

- The original fully connected layer (`model.classifier[1]`) is replaced by a new linear layer designed specifically for our two-class classification.
- This allows the model to specialize in DeepFake-specific characteristics and give us a binary output.

### 4.3.2 Xception

Similarly, for the Xception model, we apply the following modifications:

```python
model = timm.create_model('xception', pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)
```

- Here, the final fully connected layer (`fc`) is replaced by a new linear layer tailored for binary classification.
- Xception, known for its excellent performance in image-based tasks, is optimized for distinguishing real and fake images with good perfomances.

---

## 4.4 Training Procedure

After modifying the final layers, the fine-tuning training process includes the following steps:

### 4.4.1 **Data Preparation**

We create DataLoaders for training and validation datasets:

```python
train_loader = create_dataloader("processed_data/train_cropped", batch_size=32, shuffle=True)
val_loader   = create_dataloader("processed_data/val_cropped", batch_size=32, shuffle=False)
```

### 4.4.2 **Hyperparameters Definition**

The following hyperparameters are set for optimal performance:

```python
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-4
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
```

### 4.4.3 **Fine-Tuning Training Loop**

The training loop includes standard PyTorch procedures:

- Forward propagation
- Loss computation
- Backward propagation and gradient calculation
- Parameters update using Adam optimizer

The detailed code snippet:

```python
for epoch in range(EPOCHS):
    # Training
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    avg_train_loss = running_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)
    
    avg_val_loss = val_loss / total
    val_accuracy = correct / total
```

### 4.4.4 **What happens during training specifically?**

- **Forward pass**: Input images pass through pre-trained convolutional layers to extract general and specific features. The new fully connected layers transform these features into predictions.
- **Loss Calculation**: Predictions (`outputs`) are compared to ground-truth labels (`labels`) using Cross-Entropy Loss.
- **Backward Pass and Weight Update**: Gradients from the loss function propagate backward through the entire network, updating not only the newly added final layers but also fine-tuning previously learned layers to adapt better to DeepFake-specific features.

---

## 4.5 Saving the Fine-Tuned Models

After training, the models are saved for future inference and evaluation:

```python
model_save_path = f"models/{model_name}_deepfake.pth"
torch.save(model.state_dict(), model_save_path)
```

This ensures the trained parameters are stored and reusable.

---

## 4.6 Benefits and Motivations of Our Approach

We adopted a **complete fine-tuning** strategy for our pre-trained models (**MobileNet-v2** and **Xception**) because it proves to be the most effective approach for our specific task of **DeepFake detection**. 

This method allows every layer of the network to fully adapt to the unique characteristics of the OpenForensics dataset, which includes diverse and complex images containing visual anomalies typical of digital manipulations. 

Compared to retraining the last layers (feature extraction), complete fine-tuning enables the network to capture precise features, significantly enhancing generalization and robustness. 

Furthermore, having a large number of training images allows us to effectively update all model parameters,  maximizing the system's accuracy in distinguishing between real and manipulated images.
