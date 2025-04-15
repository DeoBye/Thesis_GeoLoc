import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from os.path import exists
from PIL import Image as im
from torchvision import transforms
from torch.utils.data import Dataset
import random
from PIL import ImageFilter


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
    
simple_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

augment = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])


def img_train_transform():
    train_transform_list = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    return train_transform_list

def img_val_transform():
    val_transform_list = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    return val_transform_list    


class GeoDataLoader(Dataset):
    """
    DataLoader for image-gps datasets.
    
    The expected CSV file with the dataset information should have columns:
    - 'IMG_FILE' for the image filename,
    - 'LAT' for latitude, and
    - 'LON' for longitude.
    
    Attributes:
        dataset_file (str): CSV file path containing image names and GPS coordinates.
        dataset_folder (str): Base folder where images are stored.
        transform (callable, optional): Optional transform to be applied on a sample.
    """
    source = {
        'mp16': {'img_col': 'IMG_ID', 'lat_col': 'LAT', 'lon_col':'LON', 'img_suffix': ''},
        'osv5m': {'img_col': 'file_path', 'lat_col': 'latitude', 'lon_col':'longitude', 'img_suffix': ''},
        'streetscape': {'img_col': 'uuid', 'lat_col': 'lat', 'lon_col':'lon', 'img_suffix': '.jpeg'}
    }
    def __init__(self, dataset_file, dataset_folder, source, transform=None):
        self.dataset_folder = dataset_folder
        self.transform = transform
        source_conf = self.source.get(source)
        if not source_conf:
            raise ValueError(f"not dataset loading format:{source}")
        self.img_col = source_conf['img_col']
        self.lat_col = source_conf['lat_col']
        self.lon_col = source_conf['lon_col']
        self.img_suffix = source_conf['img_suffix']
        self.images, self.coordinates = self.load_dataset(dataset_file)

    def load_dataset(self, dataset_file):
        try:
            dataset_info = pd.read_csv(dataset_file)
        except Exception as e:
            raise IOError(f"Error reading {dataset_file}: {e}")

        images = []
        coordinates = []

        for _, row in tqdm(dataset_info.iterrows(), desc="Loading image paths and coordinates"):
            filename = os.path.join(self.dataset_folder, row[self.img_col] + self.img_suffix)
            if exists(filename):
                images.append(filename)
                latitude = float(row[self.lat_col])
                longitude = float(row[self.lon_col])
                coordinates.append((latitude, longitude))

        return images, coordinates

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        #gps = self.coordinates[idx]
        gps = torch.tensor(self.coordinates[idx], dtype=torch.float32)
        
        base_pil = im.open(img_path).convert('RGB')
        
        img = simple_transform(base_pil)
        aug1 = self.transform(base_pil)
        aug2 = self.transform(base_pil)
        item = {
            'idx': idx,
            'gps': gps,
            'img': img,
            'aug1': aug1,
            'aug2': aug2
        }
        
        return item
    
class GeoDataLoader_im2gps(Dataset):
    """
    DataLoader for image-gps datasets.
    
    The expected CSV file with the dataset information should have columns:
    - 'IMG_FILE' for the image filename,
    - 'LAT' for latitude, and
    - 'LON' for longitude.
    
    Attributes:
        dataset_file (str): CSV file path containing image names and GPS coordinates.
        dataset_folder (str): Base folder where images are stored.
        transform (callable, optional): Optional transform to be applied on a sample.
    """
    def __init__(self, dataset_file, dataset_folder, transform=None):
        self.dataset_folder = dataset_folder
        self.transform = transform
        self.images, self.coordinates = self.load_dataset(dataset_file)

    def load_dataset(self, dataset_file):
        try:
            dataset_info = pd.read_csv(dataset_file)
        except Exception as e:
            raise IOError(f"Error reading {dataset_file}: {e}")

        images = []
        coordinates = []

        for _, row in tqdm(dataset_info.iterrows(), desc="Loading image paths and coordinates"):
            filename = os.path.join(self.dataset_folder, row['IMG_ID'])
            if exists(filename):
                images.append(filename)
                latitude = float(row['LAT'])
                longitude = float(row['LON'])
                coordinates.append((latitude, longitude))

        return images, coordinates

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        #gps = self.coordinates[idx]
        gps = torch.tensor(self.coordinates[idx], dtype=torch.float32)
        
        base_img = im.open(img_path).convert('RGB')
        
        img = simple_transform(base_img)
        aug1 = self.transform(base_img)
        aug2 = self.transform(base_img)
        item = {
            'idx': idx,
            'base': base_img,
            'img': img,
            'aug1': aug1,
            'aug2': aug2
        }
        
        return item
    
    
class GeoDataLoader_osv5m(Dataset):
    """
    DataLoader for image-gps datasets.
    
    The expected CSV file with the dataset information should have columns:
    - 'IMG_FILE' for the image filename,
    - 'LAT' for latitude, and
    - 'LON' for longitude.
    
    Attributes:
        dataset_file (str): CSV file path containing image names and GPS coordinates.
        dataset_folder (str): Base folder where images are stored.
        transform (callable, optional): Optional transform to be applied on a sample.
    """
    def __init__(self, dataset_file, dataset_folder, transform=None):
        self.dataset_folder = dataset_folder
        self.transform = transform
        self.images, self.coordinates = self.load_dataset(dataset_file)

    def load_dataset(self, dataset_file):
        try:
            dataset_info = pd.read_csv(dataset_file)
        except Exception as e:
            raise IOError(f"Error reading {dataset_file}: {e}")

        images = []
        coordinates = []

        for _, row in tqdm(dataset_info.iterrows(), desc="Loading image paths and coordinates"):
            filename = os.path.join(self.dataset_folder, row['file_path'])
            if exists(filename):
                images.append(filename)
                latitude = float(row['latitude'])
                longitude = float(row['longitude'])
                coordinates.append((latitude, longitude))

        return images, coordinates

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        #gps = self.coordinates[idx]
        gps = torch.tensor(self.coordinates[idx], dtype=torch.float32)
        
        image = im.open(img_path).convert('RGB')
        
        img = simple_transform(image)
        aug1 = self.transform(image)
        aug2 = self.transform(image)

        return img, gps, aug1, aug2
