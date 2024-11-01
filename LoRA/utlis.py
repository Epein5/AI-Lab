from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder

def get_data_loaders(train_dir = "../Datasets/train",val_dir ="../Datasets/val" ,batch_size = 64):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(128),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),

            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(128),
            transforms.CenterCrop(128),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456,0.406], [0.229, 0.224, 0.225])
        ])
    }

    train_dataset = ImageFolder(train_dir, transform=data_transforms['train'])
    val_dataset = ImageFolder(val_dir, transform=data_transforms['val'])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print(len(train_loader))

    return train_loader, val_loader