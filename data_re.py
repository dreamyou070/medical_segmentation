import os

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
    os.makedirs(test_img_folder, exist_ok=True)

    train_mask_folder = os.path.join(train_folder_dir, 'mask_256')
    test_mask_folder = os.path.join(test_folder_dir, 'mask_256')
    os.makedirs(test_mask_folder, exist_ok=True)

    train_mask_pil_folder = os.path.join(train_folder_dir, 'mask_pil_256')
    test_mask_pil_folder = os.path.join(test_folder_dir, 'mask_pil_256')
    os.makedirs(test_mask_pil_folder, exist_ok=True)

    train_img_files = os.listdir(train_img_folder)
    for i, file in enumerate(train_img_files):
        if i < len(train_img_files) * 0.2:
            name = file.split('.')[0]
            org_img_path = os.path.join(train_img_folder, file)
            org_mask_path = os.path.join(train_mask_folder, f'{name}.npy')
            org_mask_pil_path = os.path.join(train_mask_pil_folder, f'{name}.png')
            test_img_path = os.path.join(test_img_folder, file)
            test_mask_path = os.path.join(test_mask_folder, f'{name}.npy')
            test_mask_pil_path = os.path.join(test_mask_pil_folder, f'{name}.png')
            os.rename(org_img_path, test_img_path)
            os.rename(org_mask_path, test_mask_path)
            os.rename(org_mask_pil_path, test_mask_pil_path)