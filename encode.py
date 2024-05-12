import os
import sys
import torch
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from dataset import QualityMapDataset
from models.models import SpatiallyAdaptiveCompression
from utils import load_checkpoint, _encode

def encode_images(model, image_paths, compression_level: int, output_dir: str) -> None:
    qmap = QualityMapDataset(image_paths, mode='test', level=compression_level)
    image_paths = qmap.paths
    qmap_dataloader = DataLoader(qmap, batch_size=1, shuffle=False, num_workers=3, pin_memory=True)

    for (image, qmap), path in zip(qmap_dataloader, image_paths):
        compression_path = output_dir + '/' + str(Path(path).stem) + '.cmp'
        image = image.to(device)
        qmap = qmap.to(device)
        # _ = model(image, qmap)
        _ = _encode(model, image, compression_path, qmap)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pixelwise variable compression rate')
    parser.add_argument('-i','--image', help='path to image', type=str)
    parser.add_argument('-d','--dir', help='path to folder containing images', type=str)
    parser.add_argument('-c', '--compression', help='compression level in interval [0, 100]. Default value -100 uses uniform distribution.', default=-100, type=int)
    parser.add_argument('-o', '--out', help='output directory', type=str)
    args = parser.parse_args(sys.argv[1:])

    if args.image is None and args.dir is None:
        print("Either provide an image path or a directory path containing images.")
        parser.print_help()
        sys.exit(-1)
    
    if args.image and not os.path.isfile(args.image):
        print(f"Provided image {args.image} is not a file path")
        parser.print_help()
        sys.exit(-1)

    if args.dir and not os.path.isdir(args.dir):
        print(f"Provided directory {args.dir} is not a directory path")
        parser.print_help()
        sys.exit(-1)

    allowed_compression_values = list(range(101)) + [-100]
    if not args.compression in allowed_compression_values:
        print("Provided not supported compression level. Using default value of -100")
        args.compression = -100

    if not args.out:
        new_out_dir = os.getcwd() + "/comp"
        if not os.path.exists(new_out_dir):
            os.makedirs(new_out_dir)
        args.out = new_out_dir
        print("Provided output directory does not exist.")
        print(f"Encoded image binaries are stored in {new_out_dir}")

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    extensions = ['png', 'PNG', 'tiff', 'TIFF']
    image_files = []

    if args.image and any(e in args.image for e in extensions):
        image_files.append(args.image)
    
    if args.dir:
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
    
    checkpoint_path = "results/pretrained_dist/snapshots/2M_itrs.pt"
    device = device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SpatiallyAdaptiveCompression()
    _, model = load_checkpoint(path=checkpoint_path, model=model, device=device)
    model.eval().update()
    device = next(model.parameters()).device

    encode_images(model, image_paths, args.compression, args.out)
    
    os.remove(image_paths)