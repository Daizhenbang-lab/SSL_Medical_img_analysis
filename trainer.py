from semisupervised_finetune_simclr import SimCLRModel

import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms as T
from lightly.loss import NTXentLoss
from lightly.models.modules.heads import SimCLRProjectionHead
import os
import argparse
from data_helper import SemiSupervisedDataset
from torch.utils.data import ConcatDataset
from pytorch_lightning.callbacks import ModelCheckpoint
import pandas as pd

def main():

    parser = argparse.ArgumentParser(description='SSL finetune SimCLr')
    parser.add_argument('--dataset_path', type=str, default='dataset_size56_PSA/full_sample_PSA/train',
                        help='path to dataset')
    parser.add_argument('--num_workers', type=int, default=4, help='num workers')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--seed', type=int, default=1, help='seed')
    parser.add_argument('--max_epoch', type=int, default=50, help='max_epochs')

    args = parser.parse_args()

    pl.seed_everything(args.seed)
    model = SimCLRModel()

    for name, p in model.named_parameters():
        if "backbone" in name:
            p.requires_grad = False

    checkpoint_callback = ModelCheckpoint(
        monitor="train_total_loss",
        dirpath="checkpoints_method_lambda0.9",
        filename="simclr-{epoch:02d}-{train_total_loss:.2f}",
        save_top_k=3,
        mode="min",
    )

    gpus = torch.cuda.device_count()


    csv_path = ['dataset_size56_PSA/full_sample_PSA/train/scan122/coords_scan122.jpg.csv',
                'dataset_size56_PSA/full_sample_PSA/train/scan18/coords_scan18.jpg.csv',
                'dataset_size56_PSA/full_sample_PSA/train/scan63/coords_scan63.jpg.csv']


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

    combined_dataloader = DataLoader(
        combined_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        # collate_fn=collate_fn,
        drop_last=True,
        num_workers=args.num_workers,
        persistent_workers=True
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epoch,
        accelerator='gpu',
        devices=gpus,
        num_nodes=1,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(model, combined_dataloader)

if __name__ == '__main__':
    main()