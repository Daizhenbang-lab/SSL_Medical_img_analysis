
import argparse
import cv2

import matplotlib.pyplot as plt
from utils.Image_Preprocessing import Mask_Extraction, Extract_Sample
import os

def extract_sample_mask(args):

    slide_files = [d for d in os.listdir(args.slide_path) if os.path.isfile(os.path.join(args.slide_path, d))]
    print('Found ', len(slide_files), ' files')
    print('------------------------------------------------------------------------------')

    for i, slide_file in enumerate(slide_files):

        slide_name = os.path.basename(slide_file)
        print(f'Loading {slide_name} ...    ({i + 1})/{len(slide_files)}')

        slide_path = os.path.join(args.slide_path, slide_file)
        slide = cv2.imread(slide_path)
        #mask = Mask_Extraction(slide)
        mask_path = os.path.join(args.save_mask_path, slide_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        sample = Extract_Sample(mask, slide)


        os.makedirs(args.save_sample_path, exist_ok=True)
        # os.makedirs(args.save_mask_path, exist_ok=True)
        #
        # mask_name = os.path.join(args.save_mask_path, f'{slide_name[ :-4]}.jpg')
        sample_name = os.path.join(args.save_sample_path, f'{slide_name[ :-4]}.jpg')
        #
        # plt.imsave(mask_name, mask, cmap='gray')
        sample_rgb = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
        plt.imsave(sample_name, sample_rgb)




parser = argparse.ArgumentParser(description='Full samples and masks extraction')

parser.add_argument('--slide_path', type=str, default='WSI_PSA',
                    help='path to slide image or directory')
parser.add_argument('--save_sample_path', type=str, default='full_sample_PSA',
                    help='directory to save the sample segmentation')
parser.add_argument('--save_mask_path', type=str, default='full_mask_PSA',
                    help='directory to save the mask')

args = parser.parse_args()
extract_sample_mask(args)
print('Script finished!')