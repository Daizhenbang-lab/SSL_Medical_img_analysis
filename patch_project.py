import glob
import os, re
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import cv2

#matplotlib.use('TkAgg')

def hex_to_bgr(hex_color):

    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (rgb[2], rgb[1], rgb[0])

def plot_patches_in_image_space(csv_path, image_size, patch_size,sample_name, save_path):

    coordinates_df = pd.read_csv(csv_path)
    image_template = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

    for _, row in coordinates_df.iterrows():
        if row['sample_name'] == sample_name[ :-4]:
            top_left = (int(row['y'] - patch_size / 2), int(row['x'] - patch_size / 2))
            bottom_right = (int(row['y'] + patch_size / 2), int(row['x'] + patch_size / 2))

            color = hex_to_bgr(row['color'])
            cv2.rectangle(image_template, top_left, bottom_right, color, thickness=-1)

    image_path = os.path.join(save_path,sample_name)
    #cv2.imshow('patch project',image_template)
    cv2.imwrite( image_path , image_template)

    plt.imshow(cv2.cvtColor(image_template, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    print(f"Saved: {image_path}")


def patch_project(args):

    files = glob.glob(os.path.join(args.sample_slide, '*'))
    for i, slide in enumerate(files):
        #slide_path = os.path.join(args.sample_slide, slide)
        slide_image = Image.open(slide)
        width, height = slide_image.size

        slide_name = os.path.basename(slide)
        plot_patches_in_image_space(csv_path=args.umap_file, image_size=(width, height), patch_size=args.patch_size, sample_name = slide_name, save_path=args.save_path)
        #plot_patches_in_image_space(csv_path=args.umap_file, image_path=slide, patch_size=args.patch_size, sample_name = slide_name[ :-4])


parser = argparse.ArgumentParser(description='Patch projection')
parser.add_argument('--sample_slide', type=str, default='WSI_PSA/sample',
                        help='directory of saving original samples')
parser.add_argument('--umap_file', type=str, default='generated_result/UMAP_test_neighbor2_3D_lambda0.9.csv',
                        help='the csv file of UMAP')
parser.add_argument('--save_path', type=str, default='test_set/sample',
                        help='directory of saving patch project images')
parser.add_argument('--patch_size', type=int, default=56,
                        help='patch size')

args = parser.parse_args()
print(args)
patch_project(args)
print('Script finished!')
