import os

base_dir = r'/home/dreamyou070/MyData/anomaly_detection/medical/abdomen/abdomen_re'
image_dir = os.path.join(base_dir, 'image_512')
label_dir = os.path.join(base_dir, 'mask_512')
images = os.listdir(image_dir)

test_dir = r'/home/dreamyou070/MyData/anomaly_detection/medical/abdomen/abdomen_re/test'
os.makedirs(test_dir, exist_ok=True)
test_img_dir = os.path.join(test_dir, 'image_512')
os.makedirs(test_img_dir, exist_ok=True)
test_label_dir = os.path.join(test_dir, 'mask_512')
os.makedirs(test_label_dir, exist_ok=True)
for i, image in enumerate(images):
    if i < len(images) * 0.2:
        name, ext = os.path.splitext(image)
        label = name + '.npy'
        os.rename(os.path.join(image_dir, image), os.path.join(test_img_dir, image))
        os.rename(os.path.join(label_dir, label), os.path.join(test_label_dir, label))