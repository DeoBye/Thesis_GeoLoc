import os
import pandas as pd
from tqdm import tqdm

def add_file_paths_to_csv(csv_path, root_folder):
    """
    Add file_path column to a CSV file where each row contains the relative path
    of the image corresponding to the id.
    
    file structure for osv5m:
    osv5m:
      |___test
      |     |___01(contain images)
      |     |___02
      |     ...
      |___train
            |___01
            |___02
            ...

    before: only have img_id(file_name) without path
    updated: relative path  01/img_id, 02/img_id

    ithen we can locate image path given the path for osv5m/train


    Args:
        csv_path (str): Path to the input CSV file.
        root_folder (str): Root folder containing subfolders with images.
    """
     
    df = pd.read_csv(csv_path)

    
    id_to_path = {}

    # Traverse subfolders (00, 01, 02, ...)
    for subfolder in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, subfolder)
        if os.path.isdir(subfolder_path):
            for file_name in tqdm(os.listdir(subfolder_path)):
                if file_name.endswith('.jpg'):
                    # Extract the ID from the filename (remove .jpg)
                    file_id = int(os.path.splitext(file_name)[0])
                    # Save the relative path
                    relative_path = os.path.join(subfolder, file_name)
                    id_to_path[file_id] = relative_path

    # Map the file paths to the IDs in the CSV
    df['file_path'] = df['id'].map(id_to_path)

    # Save the updated CSV
    output_csv_path = csv_path.replace('.csv', '_with_paths.csv')
    df.to_csv(output_csv_path, index=False)
    print(f"Updated CSV saved to {output_csv_path}")

if __name__ == "__main__":
    csv_path = "/data/roof/osv5m/train.csv"  # Replace with the path to your CSV file
    root_folder = "/data/roof/osv5m/images/train"  # Replace with the root folder path containing subfolders (00, 01, ...)
    add_file_paths_to_csv(csv_path, root_folder)

                                                            
