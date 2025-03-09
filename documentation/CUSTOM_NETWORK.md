# 🏗 5. BUILDING A NETWORK FROM SCRATCH
## General description
In this section, we illustrate the structure and motivation behind the custom CNN developed for detecting DeepFake images via binary classification, distinguishing between real and fake images.

## Network architecture

### Convolutional Block 1
```PYTHON
self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
self.bn1 = nn.BatchNorm2d(32)
self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
```
- **Convolutional Layer** (`Conv2d`)
  - **Input channels**: 3 (RGB)
  - **Output channels**: 32
  - **Kernel size**: 3
  - **Padding**: 1

This initial layer is tasked with capturing and identifying low-level features, such as edges and basic colors, maintaining the spatial dimensions of the image thanks to the chosen padding, which we've set to 1.  
In this first layer, we set the number of filters used (`out_channels = 32`) to capture diverse image characteristics.

Regarding the choice of kernel, we decided to use a 3x3 size: within our specific application domain, this is well-suited for detecting small details crucial for the task of DeepFake detection.  
We also experimented with a larger 7x7 kernel, but results did not show any improvement.

- **Batch Normalization** (`BatchNorm2d`)
  - Stabilizes and accelerates training by reducing internal covariate shift, making the model more robust against variations in input distributions.  
  In fact, sudden weight changes can occur in the early layers of the network; batch normalization stabilizes training by normalizing each mini-batch, followed by scaling and shifting.

- **Max Pooling** (`MaxPool2d`)
  - **Kernel size**: 2
  - **Stride**: 2

Reduces the dimensionality of the image representation, decreasing computational load and capturing dominant features.  
Using a 2x2 kernel halves the spatial dimensions of the feature map, reducing computations without losing essential information. Indeed, we select the maximum value for each sub-matrix. Thus, this kernel size ensures significant spatial reduction while retaining the fundamental information from our images.

### Convolutional Block 2
```PYTHON
self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
self.bn2 = nn.BatchNorm2d(64)
self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
```
- **Convolutional Layer** (`Conv2d`)
  - **Input channels**: 32
  - **Output channels**: 64
  - **Kernel size**: 3
  - **Padding**: 1

Increases the network depth, allowing the capture of more complex and structured image features. Gradually increasing the number of filters enables the network to learn increasingly sophisticated representations without excessive computational overhead.

- **Batch Normalization** (`BatchNorm2d`)
  - As in the first block, it contributes to training stability and improves generalization.

- **Max Pooling** (`MaxPool2d`)
  - Maintains the dimensionality reduction approach.

### Convolutional Block 3
```PYTHON
self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
self.bn3 = nn.BatchNorm2d(128)
self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
```
- **Convolutional Layer** (`Conv2d`)
  - **Input channels**: 64
  - **Output channels**: 128
  - **Kernel size**: 3
  - **Padding**: 1

This block is crucial for acquiring high-level features, enabling the network to identify distinctive characteristics relevant to the task, such as anomalies in textures and color transitions.

- **Batch Normalization** (`BatchNorm2d`)
  - Allows for effective regularization and more stable learning.

- **Max Pooling** (`MaxPool2d`)
  - Completes the convolutional phase of the network, further reducing the spatial dimensions of feature maps, essential for minimizing overfitting.

### Fully Connected Layers
```PYTHON
self.fc1 = nn.Linear(128 * 28 * 28, 512)
self.fc2 = nn.Linear(512, num_classes)
```
- **First Fully Connected Layer** (`Linear`)
  - **Input**: 128 × 28 × 28
  - **Output**: 512

Transforms spatial representations into a compact vector representation useful for final classification. The large output dimension (512 neurons) was selected to allow the model to learn rich and informative representations.  
The input we have is a vector of dimensions 128 × 28 × 28, meaning 128 feature maps each of size 28×28 (the image): what the FC layer accomplishes is flattening, converting the 3D vector into a significantly smaller one (512 in our case).

- **Activation function**: ReLU
  - Introduces non-linearity, allowing the network to learn more complex relationships in the data compared to linear functions.

- **Dropout (0.5)**
  ```PYTHON
  self.dropout = nn.Dropout(0.5)
  ```
  - Reduces overfitting by randomly removing some neurons during training, significantly improving model generalization.  
  In our case, with a value of 0.5, we deactivate half of the neurons. This helps the network generalize better, preventing dependency on specific neurons.

- **Second Fully Connected Layer** (`Linear`)
  - **Input**: 512
  - **Output**: 2 (binary classification)

This final layer performs classification between real and deepfake images, providing the direct output required for decision-making.

## Forward function
  ```PYTHON
  def forward(self, x):
    x = self.pool1(F.relu(self.bn1(self.conv1(x))))
    x = self.pool2(F.relu(self.bn2(self.conv2(x))))
    x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
    x = torch.flatten(x, start_dim=1)
    x = F.relu(self.fc1(x))
    x = self.dropout(x)
    x = self.fc2(x)
    return x
  ```
The `forward` function defines the sequence in which data pass through the network layers:

- Input → Conv1 → BN1 → ReLU → Pool1  
- → Conv2 → BN2 → ReLU → Pool2  
- → Conv3 → BN3 → ReLU → Pool3  
- → Flatten → FC1 → ReLU → Dropout  
- → FC2 → Output  

## Technology and motivation behind choices
- **PyTorch** was chosen for its flexibility and ease of CNN implementation, as well as for its widespread use and community support.
- **Batch Normalization** significantly improves convergence speed, makes the network more robust to input variations, and allows higher learning rates.
- **Max Pooling** reduces spatial dimensions while preserving relevant features, reducing the risk of overfitting.
- **Dropout** is essential for network regularization, ensuring good generalization, particularly important for deepfake detection, where recognizing subtle and varied characteristics is crucial.

## Results obtained
We tested the network on two different datasets, **Test-Dev** and **Test-Challenge**.  
For training the network, we used the following hyperparameters and loss function:
```PYTHON
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
```
We experimented with implementing data augmentation and early stopping, but the evaluation results were unsatisfactory—worse than the baseline network—so we decided not to implement these strategies in the final training.

The results obtained on **Test-Dev** are as follows:

| Metrics  | Results |
| ------------- | ------------- |
| Accuracy  | 0.8278  |
| Precision  | 0.8918 |
| Recall   | 0.8918 |
| F1-score    | 0.9758 |

The results obtained on **Test-Challenge** are as follows:

| Metrics  | Results |
| ------------- | ------------- |
| Accuracy  | 0.8278  |
| Precision  | 0.8918 |
| Recall   | 0.8918  |
| F1-score    | 0.9758  |

We also implemented a Grad-CAM visualization to better understand the model's decision-making process, as shown below.
**Fake Image**:

![fake_gradcam_side_by_side](https://github.com/user-attachments/assets/128f4ee4-3a7a-4e8c-bebe-1d8d56c3a10a)

We can observe that the network primarily focuses its attention on the faces, particularly details and edges: it appears the model is identifying potential anomalies within the image.

**Real Image**:

![real_gradcam_side_by_side](https://github.com/user-attachments/assets/bbf226fc-fc3c-4f76-9e5c-f0d9e85a5abf)

Conversely, in the second image, representing a genuine image, the network spreads its attention more evenly, covering multiple relevant areas of the image. This suggests that the network recognizes the image as real and thus processes it more globally, rather than focusing on specific details.

