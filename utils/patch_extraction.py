import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import cv2


Image.MAX_IMAGE_PIXELS = None


def load_images(slide_path, mask_path, positive_mask_path):

    try:
        image = cv2.imread(slide_path)
    except:
        print("Could not load image")

    try:
        mask = cv2.imread(mask_path)
    except:
        print("Could not load mask")

    try:
        positive_mask = cv2.imread(positive_mask_path)
    except:
        print("Could not load mask")


    mask = np.asarray(mask) > 0
    mask = mask.astype(bool)

    positive_mask = np.asarray(positive_mask) > 0
    positive_mask = positive_mask.astype(bool)

    return image, mask, positive_mask


def zero_padding (image, patch_size):

    if image.shape[2] == 3:
        height, width, _ = image.shape
        if (height % patch_size != 0) or (width % patch_size != 0):

            new_height = (height + patch_size - 1) // patch_size * patch_size
            new_width = (width + patch_size - 1) // patch_size * patch_size

            padded_image = np.zeros((new_height, new_width, 3), dtype=image.dtype)

            pad_top = (new_height - height) // 2
            pad_left = (new_width - width) // 2

            padded_image[pad_top:pad_top + height, pad_left:pad_left + width, :] = image
        else:

            padded_image = image

    else:
        height, width = image.shape

        if (height % patch_size != 0) or (width % patch_size != 0):

            new_height = (height + patch_size - 1) // patch_size * patch_size
            new_width = (width + patch_size - 1) // patch_size * patch_size

            padded_image = np.zeros((new_height, new_width), dtype=image.dtype)
            pad_top = (new_height - height) // 2
            pad_left = (new_width - width) // 2
            padded_image[pad_top:pad_top + height, pad_left:pad_left + width: width] = image
        else:

            padded_image = image

    return padded_image


def generate_coordinates(image_shape, patch_shape, overlap):

    patch_shape = (patch_shape, patch_shape)
    stride = (patch_shape[0] * (1 - overlap), patch_shape[1] * (1 - overlap))
    xmax = image_shape[0] - patch_shape[0]
    ymax = image_shape[1] - patch_shape[1]
    coords = []
    step = np.ceil(np.divide(image_shape[ : 2], stride)).astype(np.uint32)
    x = np.ceil(np.linspace(0, xmax, step[0])).astype(np.uint32)
    y = np.ceil(np.linspace(0, ymax, step[1])).astype(np.uint32)

    for i in range(x.size):
        for j in range(y.size):
            xs = x[i]
            xe = xs + patch_shape[0]

            ys = y[j]
            ye = ys + patch_shape[1]
            coords.append([xs, xe, ys, ye])

    return coords

def save_patches(coords, mask_th, image, mask, positive_mask, save_dir,slide_name):

    os.makedirs(save_dir, exist_ok=True)
    patch_shape = (coords[0][1] - coords[0][0], coords[0][3] - coords[0][2])
    pos_x = []
    pos_y = []
    slide_names = []
    positive_pixels = []
    filenames = []
    print('Saving patches...')
    for i, indices in enumerate(tqdm(coords)):
        mask_patch = mask[indices[0]:indices[1], indices[2]:indices[3]]
        if np.mean(mask_patch) > mask_th:
            patch = image[indices[0]:indices[1], indices[2]:indices[3], :]

            positive_mask_patch = positive_mask[indices[0]:indices[1], indices[2]:indices[3]]
            positive_pixel = np.sum(positive_mask_patch != 0)
            positive_pixels.append(positive_pixel)

            pos_x.append(indices[0] + patch_shape[0] / 2)
            pos_y.append(indices[2] + patch_shape[1] / 2)
            slide_names.append(slide_name[ :-4])

            file_name = slide_name[ :-4] + '__x' + str(int(indices[0] + patch_shape[0] / 2)) + '_y' + str(
                int(indices[2] + patch_shape[1] / 2)) + '.jpg'
            filenames.append(file_name)
            save_path = os.path.join(save_dir, file_name)
            cv2.imwrite(save_path, patch)

    csv_file = os.path.join(save_dir, f'coords_{slide_name}.csv')
    df = pd.DataFrame({
        'file_name' : filenames,
        'x' : pos_x,
        'y' : pos_y,
        'positive_pixel': positive_pixels,
        'slide_name' : slide_names
    })
    df.loc[df['positive_pixel'] == 0, 'positive_pixel'] = np.nan

    if not os.path.exists(csv_file):
        df.to_csv(csv_file, index=False)

    else:

        df.to_csv(csv_file, mode='a', header=False, index=False)
    # pd.DataFrame(np.array([pos_y, pos_x]).transpose(),
    #              columns=['x', 'y']).to_csv(os.path.join(save_dir, 'coords.csv'))