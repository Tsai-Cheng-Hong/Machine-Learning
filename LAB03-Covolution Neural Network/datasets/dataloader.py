import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

torch.manual_seed(0)

# you can modify batch size here
train_batch_size = 64
test_batch_size = 2
num_workers = 0
train_size_rate = 0.8   # split dataset into train and validation 8:2

# image transform for train and test
data_transforms = transforms.Compose([
    transforms.RandomRotation(90),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
test_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# make dataset and dataloader
def make_train_dataloader(data_path):

    dataset = datasets.ImageFolder(root=data_path, transform=data_transforms)
    train_size = int(len(dataset) * train_size_rate)
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])
    
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=train_batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, valid_loader

def make_test_dataloader(data_path):

    testset = datasets.ImageFolder(root=data_path, transform=test_transforms)
    test_loader = DataLoader(testset, batch_size=test_batch_size, num_workers=num_workers)

    return test_loader