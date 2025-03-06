import torch
import torch.nn as nn
import torch.nn.functional as F

# PARAMETERS
KERNEL_SIZE = 3
PADDING = 1
POOL_KERNEL = 2
POOL_STRIDE = 2
DENSE_FEATURES = 512
N_CLASSES = 2
DROPOUT_RATE = 0.5
IMAGE_SIZE = 224

# FIRST CONVOLUTIONAL BLOCK
FIRST_IN_CHANNELS = 3
FIRST_OUT_CHANNELS = 32

# SECOND CONVOLUTIONAL BLOCK
SECOND_IN_CHANNELS = 32
SECOND_OUT_CHANNELS = 64

# THIRD CONVOLUTIONAL BLOCK
THIRD_IN_CHANNELS = 64
THIRD_OUT_CHANNELS = 128

# DENSE LAYERS
DENSE_FEATURES = 512

class CustomCNN(nn.Module):
    def __init__(self, num_classes=N_CLASSES, image_size=IMAGE_SIZE):
        super(CustomCNN, self).__init__()
        
        # Primo blocco convoluzionale
        self.conv1 = nn.Conv2d(in_channels=FIRST_IN_CHANNELS, out_channels=FIRST_OUT_CHANNELS, kernel_size=KERNEL_SIZE, padding=PADDING)
        self.bn1 = nn.BatchNorm2d(FIRST_OUT_CHANNELS)  # Stabilizzazione e accelerazione del training
        self.pool1 = nn.MaxPool2d(kernel_size=POOL_KERNEL, stride=POOL_STRIDE)
        
        # Secondo blocco convoluzionale
        self.conv2 = nn.Conv2d(in_channels=SECOND_IN_CHANNELS, out_channels=SECOND_OUT_CHANNELS, kernel_size=KERNEL_SIZE, padding=PADDING)
        self.bn2 = nn.BatchNorm2d(SECOND_OUT_CHANNELS)
        self.pool2 = nn.MaxPool2d(kernel_size=POOL_KERNEL, stride=POOL_STRIDE)
        
        # Terzo blocco convoluzionale
        self.conv3 = nn.Conv2d(in_channels=THIRD_IN_CHANNELS, out_channels=THIRD_OUT_CHANNELS, kernel_size=KERNEL_SIZE, padding=PADDING)
        self.bn3 = nn.BatchNorm2d(THIRD_OUT_CHANNELS)
        self.pool3 = nn.MaxPool2d(kernel_size=POOL_KERNEL, stride=POOL_STRIDE)
        
        # Calcolo della dimensione finale dei feature map dopo i tre pooling
        final_feature_map_size = image_size // (POOL_STRIDE ** 3)
        
        # Layer fully connected
        self.fc1 = nn.Linear(THIRD_OUT_CHANNELS * final_feature_map_size * final_feature_map_size, DENSE_FEATURES)
        self.fc2 = nn.Linear(DENSE_FEATURES, num_classes)
        
        # Dropout per ridurre l'overfitting
        self.dropout = nn.Dropout(DROPOUT_RATE)
    
    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    model = CustomCNN()
    x = torch.randn(1, FIRST_IN_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)  # Simulazione di un batch con un'immagine
    output = model(x)
    print("Output shape:", output.shape)  # Deve essere (1, 2) per la classificazione binaria