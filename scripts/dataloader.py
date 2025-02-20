import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


class DeepfakeDataset(Dataset):
    """
    Dataset PyTorch che carica i volti ritagliati (real/fake).
    Esempio di struttura cartelle:
        data/cropped_faces/
            ├── real/
            │   ├── volto1.jpg
            │   └── volto2.jpg
            └── fake/
                ├── volto3.jpg
                └── volto4.jpg
    """
    
    def __init__(self, root_dir, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        
        self.samples = []
        
        # Cartelle real e fake
        real_dir = os.path.join(root_dir, "real")
        fake_dir = os.path.join(root_dir, "fake")
        
        # Carica tutti i file di 'real'
        for fname in os.listdir(real_dir):
            if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                self.samples.append((os.path.join(real_dir, fname), 0))  # label=0 per real
        
        # Carica tutti i file di 'fake'
        for fname in os.listdir(fake_dir):
            if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                self.samples.append((os.path.join(fake_dir, fname), 1))  # label=1 per fake

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
    Crea il dataloader PyTorch per il dataset deepfake.
    
    :param root_dir: path alla cartella con /real e /fake
    :param batch_size: dimensione del batch
    :param shuffle: se mescolare i dati
    :param num_workers: numero di worker per caricare i dati
    :return: DataLoader
    """
    
    # Trasformazioni di base (esempio: resize, normalizzazione)
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
                        help="Cartella con /real e /fake")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    
    # Creiamo un dataloader di test
    loader = create_dataloader(args.data_root, batch_size=args.batch_size)
    
    for images, labels in loader:
        print("Batch di immagini:", images.shape)
        print("Batch di label:", labels)
        break  # Esempio: usciamo dal loop dopo un batch

if __name__ == "__main__":
    main()