import os
import numpy as np

base_dir = r'/home/dreamyou070/MyData/anomaly_detection/medical/brain/BraTS2020_Segmentation_256'

train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
os.makedirs(test_dir, exist_ok=True)
folders = os.listdir(train_dir)
for folder in folders:
    train_folder_dir = os.path.join(train_dir, folder)
    test_folder_dir = os.path.join(test_dir, folder)
    os.makedirs(test_folder_dir, exist_ok=True)

    train_img_folder = os.path.join(train_folder_dir, 'image_256')
    test_img_folder = os.path.join(test_folder_dir, 'image_256')

    train_mask_folder = os.path.join(train_folder_dir, 'mask_256')
    test_mask_folder = os.path.join(test_folder_dir, 'mask_256')

    train_mask_pil_folder = os.path.join(train_folder_dir, 'mask_pil_256')
    test_mask_pil_folder = os.path.join(test_folder_dir, 'mask_pil_256')

    train_mask_files = os.listdir(train_mask_folder)
    test_mask_files = os.listdir(test_mask_folder)

    for mask_file in train_mask_files:
        mask_file_path = os.path.join(train_mask_folder, mask_file)
        mask_arr = np.load(mask_file_path)
        h,w = mask_arr.shape
        if h != 256 or w != 256:
            new_mask_arr = np.zeros((256, 256), dtype=np.uint8)
            pad_h = (256 - h) // 2
            pad_w = (256 - w) // 2
            new_mask_arr[pad_h:pad_h+h, pad_w:pad_w+w] = mask_arr
        np.save(os.path.join(train_mask_folder, mask_file), new_mask_arr)


    for test_mask_file in test_mask_files :
        test_mask_file_path = os.path.join(test_mask_folder, test_mask_file)
        test_mask_arr = np.load(test_mask_file_path)
        h,w = test_mask_arr.shape
        if h != 256 or w != 256:
            new_test_mask_arr = np.zeros((256, 256), dtype=np.uint8)
            pad_h = (256 - h) // 2
            pad_w = (256 - w) // 2
            new_test_mask_arr[pad_h:pad_h+h, pad_w:pad_w+w] = test_mask_arr
        np.save(os.path.join(test_mask_folder, test_mask_file), new_test_mask_arr)
        