import lpips
import torch
import numpy as np
from basicsr.utils.registry import METRIC_REGISTRY
from basicsr.metrics.metric_util import reorder_image, to_y_channel
#
@METRIC_REGISTRY.register()
def calculate_lpips(img1, img2, crop_border=0, input_order='HWC', test_y_channel=False, net='alex'):
    """Calculate LPIPS (Learned Perceptual Image Patch Similarity).

    Args:
        img (ndarray): First image with range [0, 255].
        img2 (ndarray): Second image with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the LPIPS calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'. Default is 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.
        net (str): The network to use for LPIPS calculation. Options are 'alex', 'vgg', 'squeeze'. Default is 'alex'.

    Returns:
        float: LPIPS result.
    """
    # Reorder images
    img = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img = to_y_channel(img)
        img2 = to_y_channel(img2)

    # Convert images to PyTorch tensors
    img = img.astype(np.float32) / 255.0
    img2 = img2.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    img2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0)

    # Initialize LPIPS model
    loss_fn = lpips.LPIPS(net=net)

    # Calculate LPIPS
    lpips_value = loss_fn(img, img2)

    return lpips_value.item()


import os
import numpy as np
import torch
import lpips
from PIL import Image



def load_images_from_folder(folder):
    """Load all images from a given folder."""
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path).convert('RGB')
            img = np.array(img)
            images.append(img)
    return images

def main():
    folder1 = '/home/zrh/CUE/LOL'
    folder2 = '/home/zrh/CUE/eval15/high'

    # Load images from folders
    images1 = load_images_from_folder(folder1)
    images2 = load_images_from_folder(folder2)

    if len(images1) != len(images2):
        raise ValueError("The number of images in both folders should be the same.")

    lpips_values = []

    # Calculate LPIPS for each pair of images
    for img1, img2 in zip(images1, images2):
        lpips_value = calculate_lpips(img1, img2, crop_border=0, input_order='HWC', test_y_channel=False, net='alex')
        lpips_values.append(lpips_value)

    # Calculate average LPIPS value
    average_lpips = np.mean(lpips_values)

    # Print the result
    print(f"Average LPIPS value: {average_lpips}")

if __name__ == "__main__":
    main()
