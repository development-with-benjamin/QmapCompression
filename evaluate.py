import os
import sys
import time
import torch
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from torch.utils.data import DataLoader
from dataset import QualityMapDataset
from models.models import SpatiallyAdaptiveCompression
from utils import load_checkpoint, _encode, _decode
from torchvision.transforms.functional import to_pil_image

def psnr(original: Image, reconstructed: Image):
    original = np.array(original)
    reconstructed = np.array(reconstructed)

    sqared = (original - reconstructed)**2
    rmse = np.sqrt(np.sum(sqared) / sqared.size)
    psnr_value = 20 * np.log10(np.max(original)/rmse)
    return psnr_value

def encode_decode_images(model: SpatiallyAdaptiveCompression, image_paths: str, compression_level: int, compression_dir: str, reconstruct_dir: str) -> dict:
    qmap = QualityMapDataset(image_paths, mode='test', level=compression_level)
    image_paths = qmap.paths
    qmap_dataloader = DataLoader(qmap, batch_size=1, shuffle=False, num_workers=3, pin_memory=True)
    
    data = {'image_path': [], 'compressed_path': [], 'reconsturcted_path': [], 'bpp': [], 'psnr': [], 'compression_level': []}
    for (image, qmap), path in zip(qmap_dataloader, image_paths):
        compression_path = compression_dir + str(Path(path).stem) + '.cmp'
        reconstruct_path = reconstruct_dir + str(Path(path).stem) + '.png'

        image = image.to(device)
        qmap = qmap.to(device)
        bpp, _, _ = _encode(model, image, compression_path, qmap)
        reconstructed_image, _ = _decode(model, compression_path)

        image = image.view(image.shape[1:])
        image = to_pil_image(image)
        reconstructed_image = reconstructed_image.view(reconstructed_image.shape[1:])
        reconstructed_image = to_pil_image(reconstructed_image)
        reconstructed_image.save(reconstruct_path)
        
        psnr_value = psnr(image, reconstructed_image)

        data['bpp'].append(bpp)
        data['psnr'].append(psnr_value)
        data['image_path'].append(path)
        data['compressed_path'].append(compression_path)
        data['reconsturcted_path'].append(reconstruct_path)
        data['compression_level'].append(compression_level)
        
    return data


def imagepaths_to_csv(imagepaths: list, csv_path: str = 'tmp/compression.csv'):
    imagepaths = [str(Path(subpath).absolute()) + '\n' for subpath in imagepaths]
    imagepaths = list(set(imagepaths))

    with open(csv_path, mode='w') as csv:
        csv.write("path\n")
        csv.writelines(imagepaths)

if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser(description='Pixelwise variable compression rate')
    parser.add_argument('-d','--dir', help='path to folder containing images', type=str, required=True)
    parser.add_argument('-o', '--out', help='output directory', type=str, required=True)
    parser.add_argument('-c', "--checkpoint", help="path to checkpoint", type=str)
    args = parser.parse_args(sys.argv[1:])
    
    if not os.path.isdir(args.dir) or not os.path.isdir(args.out):
        parser.print_help()
        sys.exit(-1)
    
    compression_dir = args.out + '/comp/'
    reconstruct_dir = args.out + '/decomp/'

    imagepaths = []
    extensions = ['png', 'PNG', 'tiff', 'TIFF']
    directory, _, files = next(os.walk(args.dir))
    for file in files:
        if any(e in file for e in extensions):
            imagepaths.append(directory + '/' + file)
    tmp_file = '/tmp/compression.csv'
    imagepaths_to_csv(imagepaths, csv_path=tmp_file)

    checkpoint_path = args.checkpoint if args.checkpoint else "results/pretrained_dist/snapshots/2M_itrs.pt"
    device = device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SpatiallyAdaptiveCompression()
    _, model = load_checkpoint(path=checkpoint_path, model=model, device=device)
    model.eval().update()
    device = next(model.parameters()).device

    datafile = 'data.csv'
    pd.DataFrame(columns=['image_path', 'compressed_path', 'reconsturcted_path', 'bpp', 'psnr', 'compression_level']).to_csv(datafile)
    crompression_levels = list(range(0, 101, 5)) + [-100]
    for level in crompression_levels:
        _compression_dir = compression_dir + f'level+{str(level).zfill(3)}/' if level != -100 else compression_dir + 'level-100/'
        _reconstruct_dir = reconstruct_dir + f'level+{str(level).zfill(3)}/' if level != -100 else reconstruct_dir + 'level-100/'
        if not os.path.exists(_compression_dir):
            os.makedirs(_compression_dir)
        if not os.path.exists(_reconstruct_dir):
            os.makedirs(_reconstruct_dir)
        data = encode_decode_images(model, tmp_file, level, _compression_dir, _reconstruct_dir)
        pd.DataFrame(data).to_csv(datafile, mode='a', header=False)

    os.remove(tmp_file)
    with open('duration.txt', mode='w') as file:
        file.write(f"Last execution took {time.time() - start} seconds.")
