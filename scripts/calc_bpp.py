import os
import pandas as pd
from PIL import Image

def main() -> None:
    path = 'veindata_no_finetuning.csv'
    df = pd.read_csv(path)
    bpp = []
    for _, row in df.iterrows():
        width, height = Image.open(row['image_path']).convert('L').size
        comp_size = os.path.getsize(row['compressed_path']) * 8
        bpp.append(comp_size/(width * height))

    df['bpp'] = bpp
    df.to_csv(path)

if __name__ == '__main__':
    main()