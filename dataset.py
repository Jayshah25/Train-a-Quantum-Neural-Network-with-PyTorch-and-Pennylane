import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA





class BinaryFashionMNIST(Dataset):
    def __init__(self, class_1=0, class_2=1, train=True, n_components=4):
        # Load Fashion MNIST
        self.dataset = datasets.FashionMNIST(
            root='./data',
            train=train,
            download=True,
            transform=transforms.ToTensor()
        )
        
        # Filter for binary classes
        idx = torch.logical_or(
            self.dataset.targets == class_1,
            self.dataset.targets == class_2
        )
        self.data = self.dataset.data[idx].float() / 255.0
        self.targets = (self.dataset.targets[idx] == class_2).float()
        
        # Reshape and reduce dimensionality
        x = self.data.reshape(-1, 784).numpy()
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        
        pca = PCA(n_components=n_components)
        self.data = torch.tensor(pca.fit_transform(x_scaled)).float()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
