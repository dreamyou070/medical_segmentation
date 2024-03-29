import os
import shutil
import tempfile
import matplotlib.pyplot as plt
from tqdm import tqdm
from monai.losses import DiceCELoss

from monai.inferers import sliding_window_inference
from monai.transforms import (AsDiscrete, EnsureChannelFirstd, Compose, CropForegroundd, LoadImaged,
                              Orientationd, RandFlipd, RandCropByPosNegLabeld, RandShiftIntensityd,
                              ScaleIntensityRanged, Spacingd, RandRotate90d, )
from monai.config import print_config
from monai.metrics import DiceMetric
from monai.networks.nets import UNETR
from monai.data import (DataLoader, CacheDataset, load_decathlon_datalist, decollate_batch, )
import torch


def main():
    print(f' step 1. set data directory')
    directory = os.environ.get("MONAI_DATA_DIRECTORY")
    root_dir = tempfile.mkdtemp() if directory is None else directory  # saving directory

    print(f' step 2. Setup transforms for training and validation')
    val_transforms = Compose([LoadImaged(keys=["image", "label"]),
                              EnsureChannelFirstd(keys=["image", "label"]),
                              Orientationd(keys=["image", "label"], axcodes="RAS"),
                              Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest"), ),
                              ScaleIntensityRanged(keys=["image"],a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
                              CropForegroundd(keys=["image", "label"], source_key="image"), ])
    data_dir = r"/share0/dreamyou070/dreamyou070/MultiSegmentation/result/medical/data"
    split_json = "dataset_1.json"
    datasets = os.path.join(data_dir, split_json)
    val_files = load_decathlon_datalist(datasets, True, "validation")
    val_ds = CacheDataset(data=val_files,
                          transform=val_transforms,
                          cache_num=6, cache_rate=1.0, num_workers=4)
    #val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    print(f' step 3. Load the model')
    print(f' len of val_ds = {len(val_ds)}')
    for i in range(len(val_ds)):
        sample = val_ds.__getitem__(i)
        image = sample["image"] # arr
        label = sample["label"] # arr
        print(f'image.shape = {image.shape} | label.shape = {label.shape}')
        print(f'image = {type(image)} | label = {type(label)}')
        print(f'image max = {image.max()} | label max = {label.max()}')




if __name__ == '__main__':
    main()
