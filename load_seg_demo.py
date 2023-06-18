from mmseg.apis import init_model, inference_model, show_result_pyplot
import sys
import os
import numpy as np


if __name__ == "__main__":
    argvs = sys.argv
    if len(argvs) < 2:
        print("Expect two args: [path to img dir] and [path to mask dir]")
        
    img_dir = argvs[1]
    mask_dir = argvs[2]

    config_file = 'configs\deeplabv3plus\deeplabv3plus_r101b-d8_4xb2-80k_cityscapes-769x769.py'
    # build the model from a config file and a checkpoint file
    model = init_model(config_file)
    classes_of_cityscapes = model.dataset_meta['classes']
    print(classes_of_cityscapes)

    target_label = 'building'
    target_id = classes_of_cityscapes[classes_of_cityscapes == target_label]
    print(target_id)

    img_names = os.listdir(img_dir)
    if not os.path.exists(mask_dir):
        os.mkdir(mask_dir)

    for img_name in img_names:
        img_path = os.path.join(img_dir, img_name)
        mask_path = os.path.join(mask_dir, img_name.replace('.png', '.npy'))
        arr = np.load(mask_path)

        # mask bool

        # display the segmentation result
        # vis_image = show_result_pyplot(model, img_path, result, 1.0)