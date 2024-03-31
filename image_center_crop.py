import numpy as np
import os
from PIL import Image
def main() :

    target_size = 256
    real_size = 512
    base_folder = r'D:\medical\Abdomen\abdomen_re'
    phases = os.listdir(base_folder)
    save_folder = r'D:\medical\Abdomen\abdomen_re_256'
    os.makedirs(save_folder, exist_ok=True)
    for phase in phases:
        if 'train' in phase :
            phase_folder = os.path.join(base_folder, phase)
            save_phase_folder = os.path.join(save_folder, phase)
            os.makedirs(save_phase_folder, exist_ok=True)
            phase_folder = os.path.join(phase_folder, 'anomal')
            save_phase_folder = os.path.join(save_phase_folder, 'anomal')
            os.makedirs(save_phase_folder, exist_ok=True)
            image_folder = os.path.join(phase_folder, 'image_512')
            save_image_folder = os.path.join(save_phase_folder, 'image_256')
            os.makedirs(save_image_folder, exist_ok=True)
            mask_folder = os.path.join(phase_folder, 'mask_512')
            save_mask_folder = os.path.join(save_phase_folder, 'mask_256')
            os.makedirs(save_mask_folder, exist_ok=True)
            images = os.listdir(image_folder)
            for image in images:

                name = os.path.splitext(image)[0]
                mask_path = os.path.join(mask_folder, f'{name}.npy')
                img_path = os.path.join(image_folder, image)
                # image center crop
                np_image = np.array(Image.open(img_path))
                start = (real_size - target_size) // 2
                end = start + target_size
                np_image = np_image[start:end, start:end]
                new_pil = Image.fromarray(np_image)
                new_pil.save(os.path.join(save_image_folder, image))

                # mask center crop
                mask = np.load(mask_path)
                mask = mask[start:end, start:end]
                np.save(os.path.join(save_mask_folder, f'{name}.npy'), mask)

if __name__ == "__main__":
    main()