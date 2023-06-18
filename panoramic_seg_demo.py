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
    checkpoint_file = 'deeplabv3plus_r101b-d8_769x769_80k_cityscapes_20201226_205041-227cdf7c.pth'
    # build the model from a config file and a checkpoint file
    model = init_model(config_file, checkpoint_file, device='cuda:0')
    classes_of_cityscapes = model.dataset_meta['classes']
    print(classes_of_cityscapes)

    img_names = os.listdir(img_dir)
    if not os.path.exists(mask_dir):
        os.mkdir(mask_dir)

    for img_name in img_names:
        img_path = os.path.join(img_dir, img_name)
        mask_path = os.path.join(mask_dir, img_name.replace('.png', '.npy'))

        # inference a single image and show the result
        result = inference_model(model, img_path)
        seg_result = result.pred_sem_seg
        arr = seg_result.numpy().data
        np.save(mask_path, arr)

        # display the segmentation result
        # vis_image = show_result_pyplot(model, img_path, result, 1.0)