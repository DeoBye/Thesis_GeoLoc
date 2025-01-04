"""
Run this script to download Mapillary and KartaView images.

Input format requirement: a csv file with each row representing an image to download and
containing minimally three columns to specify its 'uuid' (the uuid assigned to the image),
'source' (whether its source is 'Mapillary' or 'KartaView'), and 'orig_id' (original ID as
given by the source).

Notes:
1. Run the script a few times until you observe no change in total number of downloaded images,
or as indicated by the message that all images have been downloaded, because not all images can
be downloaded in one go due to network issues.
2. Sometimes, despite running the script a few times until no more change is observed in the total
number of downloaded images, some images could still be missing. This is because sometimes the image
file could just be unavailable, despite presence of its metadata, due to unknown reasons (e.g.
contributor deleted the image, or maybe the image didn't pass some kind of internal quality check by
Mapillary/KartaView etc.).
"""

import pandas as pd
import os
import threading
import mapillary.interface as mly
import time
import download_jpegs_kartaview
import download_jpegs_mapillary
from pathlib import Path

def check_id(image_folder, original_csv, output_csv):
    ids = set()
    print('Checking all subfolders for existing images...')
    for subdir, dirs, files in os.walk(image_folder):
        count = 0
        for file in files:
            if file != '.DS_Store':
                ids.add(file.split('.')[0])
                count += 1
        print('Found', count, 'images in', subdir)

    # Filter out missing data and save to a new CSV file
    original_data = pd.read_csv(original_csv)
    downloaded_data = original_data[original_data['uuid'].isin(ids)]
    missing_data = original_data[~original_data['uuid'].isin(ids)]

    downloaded_data.to_csv(output_csv, index=False)
    print(f"Filtered data saved to {output_csv}. Missing images: {len(missing_data)}")
    return ids


if __name__ == '__main__':

    access_token = 'MLY|7213064342119886|26c893da27896b71623b0d5e3d3f93e5' # update your mapillary access token
    mly.set_access_token(access_token)

    # Update in_csvPath and out_jpegFolder to suit your needs
    in_csvPath = '/root/global-streetscapes/simplemaps.csv' # input csv
    out_mainFolder = './data/map_fll' # output folder to store the downloaded images
    filtered_csv =  './data/map/map_full_filter.csv'
    Path(out_mainFolder).mkdir(parents=True, exist_ok=True)

    data_l = pd.read_csv(in_csvPath).reset_index(drop=True)
    #data_l = pd.concat([data_l[data_l['source']=='Mapillary'].sample(n=25, random_state=0), data_l[data_l['source']=='KartaView'].sample(n=25, random_state=0)], ignore_index=True) # sample 50 images to download just for illustration purpose

    # increase or decrease this number to suit your need and your computer's performance
    num_thread = 100
    chunk_size = 10000000  # images will be downloaded into sub-folders with each sub-folder having maximumally 10,000 images; increase/decrease this number if you want more/fewer images per sub-folder

    already_id = check_id(out_mainFolder, in_csvPath, filtered_csv)

    print('Initiating download for new images...')

    indices = list(range(0, len(data_l), chunk_size))

    ls_df = []
    for i in range(len(indices)-1):
        start = indices[i]
        end = indices[i+1]
        df = data_l.iloc[start:end]
        ls_df.append(df)
    df = data_l.iloc[indices[-1]:]
    ls_df.append(df)

    imgcnt = 0

    for df in ls_df:

        start = df.index[0]+1
        end = df.index[-1]+1
        out_subFolder = f"{start}_{end}"
        threads = []

        index = 0

        for _, values in df.iterrows():

            uuid = values['uuid']
            # img_path = values['img_path']
            if uuid in already_id:
                continue

            if os.path.exists(os.path.join(out_mainFolder, out_subFolder)) == False:
                os.mkdir(os.path.join(out_mainFolder, out_subFolder))
            dst_path = os.path.join(
                out_mainFolder, out_subFolder, uuid + '.jpeg')
            # dst_path = os.path.join(out_mainFolder, img_path)
            image_id = values['orig_id']
            source = values['source']
            index += 1
            imgcnt += 1
            if index % num_thread == 0:
                print('Now:', imgcnt, '/', len(data_l)-len(already_id),
                      '.', 'Pre-existing:', len(already_id))
                t = threading.Thread(
                    target=download_jpegs_mapillary.download_image, args=(image_id, dst_path,))
                if source == 'KartaView':
                    t = threading.Thread(
                        target=download_jpegs_kartaview.download_image, args=(image_id, dst_path,))
                threads.append(t)
                for t in threads:
                    t.Daemon = True
                    t.start()
                t.join()
                time.sleep(0.3)
                threads = []
            else:
                t = threading.Thread(
                    target=download_jpegs_mapillary.download_image, args=(image_id, dst_path,))
                if source == 'KartaView':
                    t = threading.Thread(
                        target=download_jpegs_kartaview.download_image, args=(image_id, dst_path,))
                threads.append(t)

        print('Now:', imgcnt, '/', len(data_l)-len(already_id),
              '.', 'Pre-existing:', len(already_id))
        try:
            for t in threads:
                t.Daemon = True
                t.start()
            t.join()
        except NameError:
            print('All images for this subfolder have been downloaded.')
