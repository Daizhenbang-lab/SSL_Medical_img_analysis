import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os
import sys
from torchvision import transforms
import numpy as np
sys.path.append('/tmp/medical_image/stainlib')

from stainlib.augmentation.augmenter import HedColorAugmenter1
from utils.model import ImageCollateFunction
import torchvision.transforms as T
from lightly.transforms import RandomRotate

imagenet_normalize = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
}


class HedColorAug:
    def __init__(self, hed_thresh = 0.03):
        self.hed_thresh = hed_thresh
    def __call__(self, image):
        #print(type(image))
        dab_lighter_aug = HedColorAugmenter1(self.hed_thresh)
        dab_lighter_aug.randomize()
        return Image.fromarray(dab_lighter_aug.transform(np.array(image)))

class SemiSupervisedDataset(Dataset):
    def __init__(self, data, base_dir, transform=None,labeled=True, device='cuda'):

        self.data = data
        self.base_dir = base_dir
        self.transform = transform if transform else transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.labeled = labeled


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        row = self.data.iloc[idx]
        image_path = os.path.join(self.base_dir, row['file_name'])
        image = Image.open(image_path).convert('RGB')

        # if self.transform:
        #     image = self.transform(image)

        if self.labeled:

            image = self.transform(image)
            label = row['positive_pixel']
            return (image, image), torch.tensor(label, dtype=torch.long)
        else:

            transform = T.Compose([
                T.RandomResizedCrop(size=56,
                                    scale=(0.15, 1.0)),
                RandomRotate(prob=0.0),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.0),
                HedColorAug(hed_thresh=0.3),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            view1 = transform(image)
            view2 = transform(image)

            return (view1,view2), torch.tensor(-1, dtype=torch.long)

if __name__ == '__main__':
    # csv_path = ['/tmp/medical_image/dataset_size56_PSA/full_sample_PSA/train/scan122/coords_scan122.jpg.csv',
    #             '/tmp/medical_image/dataset_size56_PSA/full_sample_PSA/train/scan18/coords_scan18.jpg.csv',
    #             '/tmp/medical_image/dataset_size56_PSA/full_sample_PSA/train/scan63/coords_scan63.jpg.csv']

    csv_path = ['dataset_size56_PSA/full_sample_PSA/train/scan122/coords_scan122.jpg.csv',
                'dataset_size56_PSA/full_sample_PSA/train/scan18/coords_scan18.jpg.csv',
                'dataset_size56_PSA/full_sample_PSA/train/scan63/coords_scan63.jpg.csv']


    labeled_datasets = []
    unlabeled_datasets = []
    combined_datasets = []
    for csv_file in csv_path:

        base_dir = os.path.dirname(csv_file)
        df = pd.read_csv(csv_file)
        df['file_name'] = df['file_name'].astype(str)

        labeled_df = df.dropna(subset=['positive_pixel']).copy()
        unlabeled_df = df[df['positive_pixel'].isna()].copy()

        combined_datasets.append(SemiSupervisedDataset(labeled_df, base_dir, transform=None, labeled=True))
        combined_datasets.append(SemiSupervisedDataset(unlabeled_df, base_dir, transform=None, labeled=False))


    print('----------------------------------------------')
    from torch.utils.data import ConcatDataset


    combined_dataset = ConcatDataset(combined_datasets)

    num_workers = 2
    batch_size = 16
    seed = 1
    input_size = 56
    max_epochs = 25

    combined_dataloader = DataLoader(
            combined_dataset,
            batch_size=batch_size,
            shuffle=True,
            #collate_fn=collate_fn,
            drop_last=True,
            num_workers=num_workers,
            persistent_workers=True
        )
    print('-------------------------------------------------')
    for images, labels in combined_dataloader:
        print('11111111111111111111111111111111')
        print("Labeled batch - Images:", len(images), "Labels:", labels.shape)



