import os
import torch
import pandas as pd
from PIL import Image
from pytorch_msssim import ms_ssim
from torchvision import transforms

def update_dataset_with_ms_ssim(df: pd.DataFrame):
    ms_ssim_values = []
    psnr = []
    updated_df = df.copy()
    transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x.unsqueeze(0))  # Add a batch dimension
                ])
    mse = torch.nn.MSELoss()
    for i, row in df.iterrows():
        image = transform(Image.open(row['image_path']).convert('L'))
        _image = transform(Image.open(row['reconstructed_path']).convert('L'))
        x = -10 * torch.log10(1 - ms_ssim(image, _image, data_range=1.0, win_size=7))
        ms_ssim_values.append(x.item())
        x = 10 * torch.log10(1 / mse(image, _image))
        psnr.append(x.item())

    updated_df['ms_ssim'] = ms_ssim_values
    updated_df['psnr'] = psnr
    return updated_df

def update_bpp(df: pd.DataFrame) -> None:
    bpp = []
    for _, row in df.iterrows():
        width, height = Image.open(row['image_path']).convert('L').size
        comp_size = os.path.getsize(row['compressed_path']) * 8
        bpp.append(comp_size/(width * height))

    df['bpp'] = bpp
    return df

def update_csv(path: str) -> None:
    vein_data = pd.read_csv(path)
    vein_data = update_dataset_with_ms_ssim(vein_data)
    vein_data = update_bpp(vein_data)
    vein_data.to_csv(path)

def main() -> None:
    finetuned_data_path = 'veindata_with_finetuning.csv'
    not_finetuned_path = 'veindata_no_finetuning.csv'
    kodak_path = 'kodak.csv'

    update_csv(finetuned_data_path)
    update_csv(not_finetuned_path)
    update_csv(kodak_path)
    
if __name__ == '__main__':
    main()