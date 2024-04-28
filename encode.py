import os
import sys
import torch
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from dataset import QualityMapDataset
from losses.losses import Metrics, PixelwiseRateDistortionLoss
from models.models import SpatiallyAdaptiveCompression
from utils import load_checkpoint, _encode

def encode_image(image_paths, compression_level: int, output_dir: str) -> None:
    checkpoint_path = "results/pretrained_dist/snapshots/2M_itrs.pt"
    device = device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # metric = Metrics()
    # criterion = PixelwiseRateDistortionLoss()
    model = SpatiallyAdaptiveCompression()
    _, model = load_checkpoint(path=checkpoint_path, model=model, device=device)
    model.eval().update()
    device = next(model.parameters()).device

    qmap = QualityMapDataset(image_paths, mode='test', level=compression_level)
    image_paths = qmap.paths
    qmap_dataloader = DataLoader(qmap, batch_size=1, shuffle=False, num_workers=3, pin_memory=True)

    for (image, qmap), path in zip(qmap_dataloader, image_paths):
        compression_path = output_dir + '/' + str(Path(path).stem) + '.cmp'
        image = image.to(device)
        qmap = qmap.to(device)
        out_net = model(image, qmap)
        _, _, _ = _encode(model, image, compression_path, qmap)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pixelwise variable compression rate')
    parser.add_argument('-i','--image', help='path to image', type=str)
    parser.add_argument('-d','--dir', help='path to folder containing images', type=str)
    parser.add_argument('-c', '--compression', help='compression level in interval [0, 100]. Default value -100 uses uniform distribution.', default=-100, type=int)
    parser.add_argument('-o', '--out', help='output directory', type=str, required=True)
    args = parser.parse_args(sys.argv[1:])

    if not os.path.exists(args.out):
        os.makedirs(args.out)
    if not os.path.isabs(args.out):
        args.out = str(Path(args.out).absolute())

    extensions = ['png', 'jpg', 'jpeg', 'PNG', 'JPG', 'JPEG']
    image_files = []
    if args.image and os.path.isfile(args.image) and any(e in args.image for e in extensions):
        image_files.append(args.image)
    if args.dir and os.path.isdir(args.dir):
        directory, _, files = next(os.walk(args.dir))
        for file in files:
            if any(e in file for e in extensions):
                image_files.append(directory + '/' + file)
    image_files = [str(Path(subpath).absolute()) + '\n' for subpath in image_files]
    image_files = list(set(image_files))

    image_paths = '/tmp/compression.csv'
    with open(image_paths, mode='w') as csv:
        csv.write("path\n")
        csv.writelines(image_files)

    encode_image(image_paths, args.compression, args.out)