import os
import glob
import argparse
from utils.patch_extraction import load_images,generate_coordinates,save_patches, zero_padding


def extract_patches(args):


    slide_dirs = [d for d in os.listdir(args.slide_path) if os.path.isdir(os.path.join(args.slide_path, d))]
    print('Found ', len(slide_dirs), ' directories')
    print('------------------------------------------------------------------------------')

    for slide_dir in slide_dirs:
        slide_path = os.path.join(args.slide_path, slide_dir)

        mask_path = os.path.join(args.mask_path, slide_dir)
        positive_mask_path = os.path.join(args.positive_mask_path, slide_dir)

        files = glob.glob(os.path.join(slide_path, '*.jpg'))

        for i, file in enumerate(files):
            slide_name = os.path.basename(file)

            print(f'Loading {slide_name} ...    ({i + 1})/{len(files)}')
            mask_file = os.path.join(mask_path, slide_name)
            positive_mask_file = os.path.join(positive_mask_path, slide_name)

            if not os.path.exists(mask_file):
                print(f'Mask file for {slide_name} not found, skipping...')
                continue

            if not os.path.exists(positive_mask_file):
                print(f'Positive mask file for {slide_name} not found, skipping...')
                continue

            image, mask, positive_mask = load_images(file, mask_file, positive_mask_file)
            if image is None or mask is None or positive_mask is None:
                print(f'Failed to load image or mask for {slide_name}, skipping...')
                continue

            image = zero_padding(image, args.patch_shape)
            mask = zero_padding(mask, args.patch_shape)
            positive_mask = zero_padding(positive_mask, args.patch_shape)

            coords = generate_coordinates(mask.shape, args.patch_shape, args.overlap)

            subpath = os.path.join(slide_path, slide_name[:-4])
            save_path = os.path.join(args.save_path, subpath)
            os.makedirs(save_path, exist_ok=True)
            save_patches(coords, args.mask_th, image, mask, positive_mask, save_path, slide_name)

            print(f'Saved patches for {slide_name}     ({i + 1})/{len(files)}')
            print('------------------------------------------------------------------------------')


parser = argparse.ArgumentParser(description='Patch and feature extraction configuration')
parser.add_argument('--slide_path', type=str, default='full_sample_PSA',
                    help='path to slide image or directory')
parser.add_argument('--mask_path', type=str, default='full_mask_PSA',
                    help='path to mask image or directory')
parser.add_argument('--positive_mask_path', type=str, default='mask_PSA_Positive',
                    help='path to mask image or directory')

parser.add_argument('--patch_shape', type=int, default=56,
                    help='desired shape of the patches')
parser.add_argument('--overlap', type=float, default=.5,
                    help='percentage of overlap between patches (0-1)')
parser.add_argument('--save_path', type=str, default='dataset_size56_PSA',
                    help='directory to save the extracted patches')
parser.add_argument('--mask_th', type=float, default=.6,
                    help='minimum percentage of mask to accept a patch (0-1)')


args = parser.parse_args()
extract_patches(args)
print('Script finished!')