import os
import torch
import pandas as pd
from pathlib import Path
from PIL import Image
from models.models import SpatiallyAdaptiveCompression
from torchvision.transforms.functional import to_pil_image
from utils import load_checkpoint, _decode

def bpp(image: Image, compression_path: str) -> float:
    size = os.path.getsize(compression_path)
    _, _, width, height = image.

if __name__ == "__main__":
    data = pd.read_csv("veindata_with_finetuning.csv")
    compressed_files = data['compressed_path'].to_list()
    bpp_values = data['bpp'].to_list()

    checkpoint_path = "results/veins/snapshots/best.p"
    model = SpatiallyAdaptiveCompression()
    _, model = load_checkpoint(path=checkpoint_path, model=model, device=device)

    pass