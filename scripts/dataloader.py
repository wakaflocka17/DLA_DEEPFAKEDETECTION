import os
from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


class DeepfakeDataset(Dataset):
    """
    PyTorch dataset that loads cropped faces (real/fake).
    Example of folder structure:
        processed_data/Train/
            ├── real/
            │   ├── face1.jpg
            │   └── face2.jpg
            └── fake/
                ├── face3.jpg
                └── face4.jpg
    """
    
    def __init__(self, root_dir, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        
        self.samples = []
        
        # Real and fake folders
        real_dir = os.path.join(root_dir, "real")
        fake_dir = os.path.join(root_dir, "fake")
        
        if not os.path.exists(real_dir) or not os.path.exists(fake_dir):
            raise ValueError(f"Error: The directories '{real_dir}' and/or '{fake_dir}' do not exist!")

        # Load all files from 'real'
        for fname in os.listdir(real_dir):
            if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                self.samples.append((os.path.join(real_dir, fname), 0))  # label=0 for real
        
        # Load all files from 'fake'
        for fname in os.listdir(fake_dir):
            if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                self.samples.append((os.path.join(fake_dir, fname), 1))  # label=1 for fake

        if len(self.samples) == 0:
            raise ValueError(f"Error: No images found in {real_dir} or {fake_dir}!")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert("RGB")
        except UnidentifiedImageError:
            print(f"Warning: Unable to open {img_path}. Skipping...")
            return self.__getitem__((idx + 1) % len(self.samples))  # Skip corrupted image

        if self.transform is not None:
            image = self.transform(image)
        
        return image, label


def create_dataloader(root_dir, batch_size=32, shuffle=True, num_workers=0):
    """
    Creates a PyTorch DataLoader for the deepfake dataset.
    
    :param root_dir: path to the folder containing /real and /fake
    :param batch_size: batch size
    :param shuffle: whether to shuffle the data
    :param num_workers: number of workers for data loading
    :return: DataLoader
    """
    
    # Basic transformations (example: resize, normalization)
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    
    dataset = DeepfakeDataset(root_dir=root_dir, transform=transform)
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["Train", "Val", "Test-Dev", "Test-Challenge"], required=True,
                        help="Dataset to load: 'Train', 'Val', 'Test-Dev', 'Test-Challenge'")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    args = parser.parse_args()
    
    data_root = f"processed_data/{args.dataset}"
    
    # Create a test dataloader
    loader = create_dataloader(data_root, batch_size=args.batch_size)
    
    for images, labels in loader:
        print(f"Loaded batch from {args.dataset} → Images: {images.shape}, Labels: {labels}")
        break  # Example: exit the loop after one batch

if __name__ == "__main__":
    main()