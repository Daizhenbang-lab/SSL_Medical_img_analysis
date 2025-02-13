import os, re

import cv2
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import dataloader
import torchvision

import lightly.data as data
from sklearn.preprocessing import normalize
from tqdm import tqdm
import umap.umap_ as umap
from matplotlib.colors import rgb2hex
import pickle
import anndata
from utils.model import SimCLRModel
from utils.patch_extraction import zero_padding

def create_dataloader(patch_path, num_workers):

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=data.collate.imagenet_normalize['mean'],
            std=data.collate.imagenet_normalize['std'],
        )
    ])

    dataset = data.LightlyDataset(
        input_dir=patch_path,
        transform=transforms
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        persistent_workers=True
    )

    return dataloader

def load_model_weights(model, weights):
    model_dict = model.state_dict()
    weights = {k: v for k, v in weights.items() if k in model_dict}
    if weights == {}:
        print('No weight could be loaded..')
    model_dict.update(weights)
    model.load_state_dict(model_dict)
    return model

def extract_features(model, dataloader):
    embeddings = []
    filenames = []
    with torch.no_grad():
        print('Extracting features...')
        for img, label, fnames in tqdm(dataloader):
            # img = img.to(model.device)
            emb = model(img).flatten(start_dim=1)
            embeddings.append(emb)
            filenames.extend(fnames)

    embeddings = torch.cat(embeddings, 0)
    #embeddings = normalize(embeddings)
    return embeddings, filenames


def load_model(model_path, architecture, imagenet_weights=False):

    if 'simclr' in model_path:
        model = SimCLRModel()
    else:
        model = torchvision.models.__dict__[architecture](pretrained=imagenet_weights)
    if imagenet_weights == False:
        print('Loading model from ', model_path)
        state = torch.load(model_path, weights_only=True)
        state_dict = state['state_dict']
        for key in list(state_dict.keys()):
            state_dict[key.replace('model.', '').replace('resnet.', '')] = state_dict.pop(key)
        model = load_model_weights(model, state_dict)

    if 'simclr' in model_path:
        backbone = nn.Sequential(*list(model.children())[:-2])
    else:
        backbone = nn.Sequential(*list(model.children())[:-1])

    return backbone



def generate_umap(patch_path, experiment_name, embeddings, filenames, sample_name, quarter_path):

    pickle.dump(embeddings, open(os.path.join(patch_path, 'embeddings_' + experiment_name + '.p'), "wb"))
    #pickle.dump(filenames, open(os.path.join(patch_path, 'filenames_' + experiment_name + '.p'), "wb"))

    #reduce the dimention
    reducer = umap.UMAP(n_components=3)
    components = reducer.fit_transform(embeddings)


    rgb_feat = (components - components.min(axis=0)) / (components.max(axis=0) - components.min(axis=0))
    hex_feat = [rgb2hex(rgb_feat[i, :]) for i in range(len(rgb_feat))]
    hex_feat = np.reshape(np.array(hex_feat), (len(hex_feat), 1))

    pos_y = []
    pos_x = []
    for filename in filenames:

        if filename.endswith(".jpg"):

            y_match = re.search(r'y(\d+)', filename)
            if not y_match:
                raise ValueError(f"No 'y' value found in filename: {filename}")
            y = int(y_match.group(1))
            pos_y.append(y)

            x_match = re.search(r'x(\d+)', filename)
            if not x_match:
                raise ValueError(f"No 'x' value found in filename: {filename}")
            x = int(x_match.group(1))
            pos_x.append(x)

    df = pd.DataFrame(np.hstack([np.array([components[:, 0], components[:, 1],components[:, 2], pos_x, pos_y]).transpose(), hex_feat]),
                      columns=['UMAP1', 'UMAP2', 'UMAP3', 'x', 'y', 'color'])

    df['sample_name'] = [s.split('/') [1] for s in sample_name]

    csv_file = os.path.join(patch_path, f'UMAP_{experiment_name}.csv')
    df.to_csv(csv_file)


def csv2h5ad(patch_path, experiment_name):

    coordinates_df = pd.read_csv(os.path.join(patch_path, 'UMAP_'+experiment_name+'.csv'))
    with open(os.path.join(patch_path, 'embeddings_'+experiment_name+'.p'), "rb") as f:
        features_array = pickle.load(f)

    obs_df = coordinates_df[["color"]]
    UMAP_array = coordinates_df[["UMAP1","UMAP2","UMAP3"]].to_numpy()
    spatial_array = coordinates_df[["x","y"]].to_numpy()

    obsm_df = {}
    obsm_df["X_umap"] = UMAP_array
    obsm_df["spatial"] = spatial_array
    obsm_df["features"] = features_array

    adata = anndata.AnnData(None, obs=obs_df, obsm=obsm_df)
    adata.write_h5ad(os.path.join(patch_path, experiment_name+'.h5ad'))