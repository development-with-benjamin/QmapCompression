import os
import sys
import torch
import argparse
from pathlib import Path
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from models.models import SpatiallyAdaptiveCompression
from utils import load_checkpoint, _decode


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pixelwise variable compression rate')
    parser.add_argument('-c','--comp', help='path to compressed binary file', type=str)
    parser.add_argument('-d','--dir', help='path to folder containing compressed binary files', type=str)
    parser.add_argument('-o', '--out', help='output directory', type=str, required=True)
    args = parser.parse_args(sys.argv[1:])

    if not os.path.exists(args.out):
        os.makedirs(args.out)
    if not os.path.isabs(args.out):
        args.out = str(Path(args.out).absolute())

    compressed_files = []
    if args.comp and os.path.isfile(args.comp) and args.comp.endswith('.cmp'):
        compressed_files.append(args.comp)
    if args.dir and os.path.isdir(args.dir):
        directory, _, files = next(os.walk(args.dir))
        compressed_files += [directory + '/' + file for file in files if file.endswith('.cmp')]

    checkpoint_path = "results/pretrained_dist/snapshots/2M_itrs.pt"
    device = device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SpatiallyAdaptiveCompression()
    _, model = load_checkpoint(path=checkpoint_path, model=model, device=device)
    model.eval().update()
    device = next(model.parameters()).device

    for filepath in compressed_files:
        filename = str(Path(filepath).stem)
        reconstructed_image, _ = _decode(model, filepath)
        reconstructed_image = reconstructed_image.view(reconstructed_image.shape[1:])
        reconstructed_image = to_pil_image(reconstructed_image)
        reconstructed_image.save(args.out + '/' + filename + '.png')