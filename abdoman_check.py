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
                              ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0,
                                                   clip=True),
                              CropForegroundd(keys=["image", "label"], source_key="image"), ])

    data_dir = r"share0/dreamyou070/dreamyou070/MultiSegmentation/result/medical/data"
    split_json = "dataset_0.json"
    datasets = os.path.join(data_dir, split_json)
    val_files = load_decathlon_datalist(datasets, True, "validation")
    val_ds = CacheDataset(data=val_files,
                          transform=val_transforms,
                          cache_num=6, cache_rate=1.0, num_workers=4)
    """
    print(f' step 3. Check data shape and visualize')
    slice_map = {"img0035.nii.gz": 170,
        "img0036.nii.gz": 230,
        "img0037.nii.gz": 204,
        "img0038.nii.gz": 204,
        "img0039.nii.gz": 204,
        "img0040.nii.gz": 180,
    }
    case_num = 0
    img_name = os.path.split(val_ds[case_num]["image"].meta["filename_or_obj"])[1]
    """

    """


    datalist = load_decathlon_datalist(datasets, True, "training") # image and label directory

    print(f'--------------------------------------------------------------')
    print(datalist)
    print(f'--------------------------------------------------------------')
    val_files = load_decathlon_datalist(datasets, True, "validation")

    train_ds = CacheDataset(data=datalist,
                            transform=train_transforms,
                            cache_num=24,
                            cache_rate=1.0,
                            num_workers=8,
                            reader = 'NibabelReade')
    """
    """
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_num=6, cache_rate=1.0, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    slice_map = {
        "img0035.nii.gz": 170,
        "img0036.nii.gz": 230,
        "img0037.nii.gz": 204,
        "img0038.nii.gz": 204,
        "img0039.nii.gz": 204,
        "img0040.nii.gz": 180,
    }

    case_num = 0
    img_name = os.path.split(val_ds[case_num]["image"].meta["filename_or_obj"])[1]
    img = val_ds[case_num]["image"]
    label = val_ds[case_num]["label"]
    img_shape = img.shape
    label_shape = label.shape
    print(f"image shape: {img_shape}, label shape: {label_shape}")
    plt.figure("image", (18, 6))
    plt.subplot(1, 2, 1)
    plt.title("image")
    plt.imshow(img[0, :, :, slice_map[img_name]].detach().cpu(), cmap="gray")
    plt.subplot(1, 2, 2)
    plt.title("label")
    plt.imshow(label[0, :, :, slice_map[img_name]].detach().cpu())
    plt.show()
    """


if __name__ == '__main__':
    main()
