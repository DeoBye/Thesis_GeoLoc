import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image as im
from os.path import exists, basename, splitext
from glob import glob

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import CLIPModel, AutoProcessor

def img_val_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

class GeoDataLoader(Dataset):
    source = {
        'mp16': {'img_col': 'IMG_ID', 'lat_col': 'LAT', 'lon_col':'LON', 'img_suffix': ''},
        'osv5m': {'img_col': 'file_path', 'lat_col': 'latitude', 'lon_col':'longitude', 'img_suffix': ''},
        'streetscape': {'img_col': 'uuid', 'lat_col': 'lat', 'lon_col':'lon', 'img_suffix': '.jpeg'}
    }

    def __init__(self, dataset_file, dataset_folder, source='osv5m', transform=None):
        self.dataset_folder = dataset_folder
        self.transform = transform
        conf = self.source.get(source)
        if not conf:
            raise ValueError(f"Unsupported source: {source}")
        self.img_col = conf['img_col']
        self.lat_col = conf['lat_col']
        self.lon_col = conf['lon_col']
        self.img_suffix = conf['img_suffix']
        self.images, self.coordinates = self.load_dataset(dataset_file)

    def load_dataset(self, dataset_file):
        dataset_info = pd.read_csv(dataset_file)
        images, coordinates = [], []
        for _, row in tqdm(dataset_info.iterrows(), desc=f"Loading {dataset_file}"):
            path = os.path.join(self.dataset_folder, row[self.img_col] + self.img_suffix)
            if exists(path):
                images.append(path)
                coordinates.append((float(row[self.lat_col]), float(row[self.lon_col])))
        return images, coordinates

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        gps = torch.tensor(self.coordinates[idx], dtype=torch.float32)
        image = im.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return img_path, image, gps

class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.CLIP = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
        for p in self.CLIP.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.CLIP.get_image_features(pixel_values=x)

def process_and_save_embeddings(dataset_file, dataset_folder, output_file, batch_size=64, device='cuda:0', source='osv5m'):
    if exists(output_file):
        print(f"⏩ Skipping {output_file}, already exists.")
        return

    transform = img_val_transform()
    dataset = GeoDataLoader(dataset_file, dataset_folder, source, transform)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)

    encoder = ImageEncoder().to(device).eval()
    all_filenames, all_embeddings, all_coordinates = [], [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Embedding {basename(dataset_file)}"):
            batch = [item for item in batch if item is not None]
            if len(batch) == 0:
                continue

            filenames, images, coords = zip(*batch)
            images = torch.stack(images).to(device)
            coords = torch.stack(coords).to(device)

            embeddings = encoder(images).cpu().numpy()
            coords_np = coords.cpu().numpy()

            all_filenames.extend(filenames)
            all_embeddings.append(embeddings)
            all_coordinates.append(coords_np)

    np.savez(output_file,
             filenames=np.array(all_filenames),
             embeddings=np.concatenate(all_embeddings, axis=0),
             coordinates=np.concatenate(all_coordinates, axis=0))
    print(f"✅ Saved {len(all_filenames)} samples to {output_file}")

def merge_npz_files(npz_folder, output_file):
    all_embeddings, all_coordinates, all_filenames = [], [], []
    for file in sorted(glob(f"{npz_folder}/*.npz")):
        data = np.load(file)
        all_embeddings.append(data['embeddings'])
        all_coordinates.append(data['coordinates'])
        all_filenames.extend(data['filenames'])

    np.savez(output_file,
             filenames=np.array(all_filenames),
             embeddings=np.concatenate(all_embeddings),
             coordinates=np.concatenate(all_coordinates))
    print(f"🧩 Merged to {output_file}")

def split_csv(dataset_file, out_dir, chunk_size):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(dataset_file)
    for i, start in enumerate(range(0, len(df), chunk_size)):
        df.iloc[start:start + chunk_size].to_csv(os.path.join(out_dir, f"part_{i}.csv"), index=False)
    print(f"✅ Split into {len(df) // chunk_size + 1} parts in {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default="/root/data/osv5m/train_with_paths.csv", help="Path to full CSV")
    parser.add_argument('--split_dir', type=str, default="./splits")
    parser.add_argument('--dataset_folder', type=str, default="/root/data/osv5m/images/train")
    parser.add_argument('--npz_dir', type=str, default="./npz_outputs")
    parser.add_argument('--source', type=str, default='osv5m')
    parser.add_argument('--chunk_size', type=int, default=50000)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--merge', action='store_true', help="Merge all .npz into one")
    parser.add_argument('--merged_out', type=str, default="final_embeddings.npz")
    args = parser.parse_args()

    if args.csv:
        split_csv(args.csv, args.split_dir, args.chunk_size)

    os.makedirs(args.npz_dir, exist_ok=True)
    for csv_file in sorted(os.listdir(args.split_dir)):
        input_path = os.path.join(args.split_dir, csv_file)
        part_name = splitext(csv_file)[0]
        output_path = os.path.join(args.npz_dir, f"{part_name}.npz")
        process_and_save_embeddings(input_path, args.dataset_folder, output_path,
                                    batch_size=args.batch_size, device=args.device, source=args.source)

    if args.merge:
        merge_npz_files(args.npz_dir, args.merged_out)
