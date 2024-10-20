import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from archi import Drisciminator, Generator

class Config:
    def __init__(self):
        # Hyperparameters
        self.lr = 1e-4
        self.z_dim = 64
        self.image_dim = 28 * 28 * 1
        self.batch_size = 32
        self.epochs = 50
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize models
        self.disc = Drisciminator(self.image_dim).to(self.device)
        self.gen = Generator(self.z_dim, self.image_dim).to(self.device)
        
        # Fixed noise for testing
        self.fixed_noise = torch.randn(self.batch_size, self.z_dim).to(self.device)
        
        # Data transforms
        self.transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
        
        # Dataset and loader
        self.dataset = datasets.MNIST(
            root="datasets/", 
            transform=self.transforms, 
            download=True
        )
        self.loader = DataLoader(
            self.dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        
        # Optimizers
        self.optim_disc = optim.Adam(self.disc.parameters(), lr=self.lr)
        self.optim_gen = optim.Adam(self.gen.parameters(), lr=self.lr)
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # TensorBoard writers
        self.writer_fake = SummaryWriter(f"runs/GAN_MNIST/fake")
        self.writer_real = SummaryWriter(f"runs/GAN_MNIST/real")

