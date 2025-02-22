import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


class DeepfakeDataset(Dataset):
    """
    PyTorch dataset that loads cropped faces (real/fake).
    Example of folder structure:
        data/cropped_faces/
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
        
        # Load all files from 'real'
        for fname in os.listdir(real_dir):
            if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                self.samples.append((os.path.join(real_dir, fname), 0))  # label=0 for real
        
        # Load all files from 'fake'
        for fname in os.listdir(fake_dir):
            if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                self.samples.append((os.path.join(fake_dir, fname), 1))  # label=1 for fake

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        image = Image.open(img_path).convert("RGB")
        
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
    parser.add_argument("--data_root", type=str, default="processed_data",
                        help="Folder containing /real and /fake")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    
    # Create a test dataloader
    loader = create_dataloader(args.data_root, batch_size=args.batch_size)
    
    for images, labels in loader:
        print("Batch of images:", images.shape)
        print("Batch of labels:", labels)
        break  # Example: exit the loop after one batch

if __name__ == "__main__":
    main()