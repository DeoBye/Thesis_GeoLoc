import os
import zipfile
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def unzip_and_remove(file_path):
    """ extract images from downloaded osv5m zip file and delele raw zip file"""
    root = os.path.dirname(file_path)
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(root)
    os.remove(file_path)

def main():
    """parallel process all zip file"""
    zip_files = []
    for root, dirs, files in os.walk("/data/roof/osv/osvtest"):
        for file in files:
            if file.endswith(".zip"):
                zip_files.append(os.path.join(root, file))
    
    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(unzip_and_remove, zip_files), total=len(zip_files), desc="Extracting ZIP files"))

if __name__ == "__main__":
    main()
