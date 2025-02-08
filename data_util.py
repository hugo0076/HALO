import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
import pickle

class CIFAR10WithExtra(Dataset):
    def __init__(self, split='train', train_transform=None, eval_transform=None, extra_frac=0.7, aux_path=None):
        if split not in ['train', 'val', 'test']:
            raise ValueError("split must be one of 'train', 'val', or 'test'")
        
        self.split = split
        self.train_transform = train_transform
        self.eval_transform = eval_transform
        self.extra_frac = extra_frac

        # Load CIFAR10
        if split == 'test':
            self.cifar = datasets.CIFAR10('data', train=False, download=True)
        else:
            full_train = datasets.CIFAR10('data', train=True, download=True)
            labels = np.array(full_train.targets)
            train_indices, val_indices = [], []
            for class_id in range(10):
                class_indices = np.where(labels == class_id)[0]
                np.random.shuffle(class_indices)
                train_indices.extend(class_indices[:-500])
                val_indices.extend(class_indices[-500:])
            
            if split == 'train':
                self.cifar = torch.utils.data.Subset(full_train, train_indices)
            else:  # val
                self.cifar = torch.utils.data.Subset(full_train, val_indices)

        self.n_cifar = len(self.cifar)
        
        # Load extra data if needed
        if extra_frac > 0 and aux_path and split == 'train':
            with open(aux_path, 'rb') as f:
                aux = pickle.load(f)
            self.aux_data = aux['data']
            self.aux_targets = aux['extrapolated_targets']
            self.n_extra = len(self.aux_data)
            self.extra_per_epoch = int(self.n_cifar * extra_frac / (1 - extra_frac))
            self.total_length = self.n_cifar + self.extra_per_epoch
        else:
            self.total_length = self.n_cifar

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        if idx < self.n_cifar:
            img, target = self.cifar[idx]
        else:
            extra_idx = idx - self.n_cifar
            img = self.aux_data[extra_idx]
            target = self.aux_targets[extra_idx]
        
        if self.split == 'train':
            if self.train_transform:
                img = self.train_transform(img)
        else:
            if self.eval_transform:
                img = self.eval_transform(img)
        
        return img, target

class MixedSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
        self.n_cifar = data_source.n_cifar
        self.extra_per_epoch = getattr(data_source, 'extra_per_epoch', 0)
        self.n_extra = getattr(data_source, 'n_extra', 0)

    def __iter__(self):
        cifar_indices = torch.randperm(self.n_cifar).tolist()
        if self.extra_per_epoch > 0:
            extra_indices = (torch.randperm(self.n_extra)[:self.extra_per_epoch] + self.n_cifar).tolist()
            indices = cifar_indices + extra_indices
            np.random.shuffle(indices)
        else:
            indices = cifar_indices
        return iter(indices)

    def __len__(self):
        return len(self.data_source)

def get_cifar10_loader(batch_size, split='train', train_transform=None, eval_transform=None, extra_frac=0, aux_path=None):
    dataset = CIFAR10WithExtra(split, train_transform, eval_transform, extra_frac, aux_path)
    if split == 'train':
        sampler = MixedSampler(dataset)
        shuffle = False
    else:
        sampler = None
        shuffle = False
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=shuffle, num_workers=1)

def get_datasets(dataset_name, batch_size, oe_dataset=None, extra_fraction = 0.7, verbose=False):
    dataset_name = dataset_name.upper()

    # Define transforms
    if dataset_name == "TINYIMAGENET":
        train_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
    else:  # CIFAR10 and CIFAR100
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    # Load datasets
    if dataset_name == "CIFAR10S":
        train_loader = get_cifar10_loader(batch_size=batch_size, split='train', 
                                train_transform=train_transform, eval_transform=transforms.ToTensor(),
                                extra_frac=extra_fraction, aux_path='data/ti_500K_pseudo_labeled.pickle')
        val_loader = get_cifar10_loader(batch_size=batch_size, split='val', 
                                        train_transform=train_transform, eval_transform=transforms.ToTensor())
        test_loader = get_cifar10_loader(batch_size=batch_size, split='test', 
                                        train_transform=train_transform, eval_transform=transforms.ToTensor())
        print("CIFAR10S with extra data")
        print("Train: ", len(train_loader.dataset))
        print("Val: ", len(val_loader.dataset))
        print("Test: ", len(test_loader.dataset))
    else:
        if dataset_name == "CIFAR10":
            train_data = datasets.CIFAR10(root="data", train=True, download=True, transform=train_transform)
            test_data = datasets.CIFAR10(root="data", train=False, download=True, transform=test_transform)
        elif dataset_name == "CIFAR100":
            train_data = datasets.CIFAR100(root="data", train=True, download=True, transform=train_transform)
            test_data = datasets.CIFAR100(root="data", train=False, download=True, transform=test_transform)
        elif dataset_name == "TINYIMAGENET":
            train_data = datasets.ImageFolder("data/tiny-imagenet-200/train", transform=train_transform)
            test_data = datasets.ImageFolder("data/tiny-imagenet-200/val", transform=test_transform)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        # Split train data into train and validation
        train_size = int(0.9 * len(train_data))
        val_size = len(train_data) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(train_data, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Create OE dataloader if needed (unchanged)
    oe_loader = None
    if oe_dataset:
        oe_dataset = oe_dataset.upper()
        if oe_dataset == "TIN597":
            oe_transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
            oe_dataset = datasets.ImageFolder("data/tin597/test", transform=oe_transform)
            oe_loader = DataLoader(oe_dataset, batch_size=batch_size, shuffle=True)
        else:
            raise ValueError(f"Unsupported OE dataset: {oe_dataset}")

    if verbose:
        print(f"Dataset: {dataset_name}")
        print(f"Train set size: {len(train_loader.dataset)}")
        print(f"Validation set size: {len(val_loader.dataset)}")
        print(f"Test set size: {len(test_loader.dataset)}")
        if oe_loader:
            print(f"OE dataset: {oe_dataset}")
            print(f"OE set size: {len(oe_loader.dataset)}")

    return train_loader, val_loader, test_loader, oe_loader