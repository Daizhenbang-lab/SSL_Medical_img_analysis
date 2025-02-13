import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
from lightly.data import collate, LightlyDataset
from PIL import Image
import numpy as np
from lightly.models.modules.heads import SimCLRProjectionHead
from lightly.loss import NTXentLoss
from torchvision.transforms.functional import to_tensor
import sys
from pytorch_lightning.strategies.ddp import DDPStrategy
#sys.path.append('/tmp/pycharm_project_794/stainlib')

from stainlib.augmentation.augmenter import HedColorAugmenter1
import torchvision.transforms as T
import os
from pytorch_lightning.callbacks import ModelCheckpoint
from lightly.transforms import RandomRotate

path_to_model = 'tenpercent_resnet18.ckpt'
max_epochs = 25

class HedColorAug:
    def __init__(self, hed_thresh = 0.03):
        self.hed_thresh = hed_thresh
    def __call__(self, image):
        #print(type(image))
        dab_lighter_aug = HedColorAugmenter1(self.hed_thresh)
        dab_lighter_aug.randomize()
        return Image.fromarray(dab_lighter_aug.transform(np.array(image)))

imagenet_normalize = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
}

class ImageCollateFunction(collate.BaseCollateFunction):
    def __init__(self,
                 input_size: int = 64,
                 min_scale: float = 0.15,
                 vf_prob: float = 0.0,
                 hf_prob: float = 0.5,
                 rr_prob: float = 0.0,
                 hed_thresh: float = 0.3,
                 normalize: dict = imagenet_normalize):

        if isinstance(input_size, tuple):
            input_size = max(input_size)
        else:
            input_size = input_size

        transform = [T.RandomResizedCrop(size=input_size,
                                         scale=(min_scale, 1.0)),
                     RandomRotate(prob=rr_prob),
                     T.RandomHorizontalFlip(p=hf_prob),
                     T.RandomVerticalFlip(p=vf_prob),
                     HedColorAug(hed_thresh=hed_thresh),
                     T.ToTensor()
                     ]

        if normalize:
            transform += [
                T.Normalize(
                    mean=normalize['mean'],
                    std=normalize['std'])
            ]

        transform = T.Compose(transform)

        super(ImageCollateFunction, self).__init__(transform)


def load_model_weights(model, weights):
    model_dict = model.state_dict()
    weights = {k: v for k, v in weights.items() if k in model_dict}
    if weights == {}:
        print('No weight could be loaded..')
    model_dict.update(weights)
    model.load_state_dict(model_dict)
    return model

class SimCLRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

        resnet = torchvision.models.__dict__['resnet18'](pretrained=False)
        state = torch.load(path_to_model)
        state_dict = state['state_dict']
        for key in list(state_dict.keys()):
            state_dict[key.replace('model.', '').replace('resnet.', '')] = state_dict.pop(key)
        resnet = load_model_weights(resnet, state_dict)

        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        print(self.backbone)
        hidden_dim = resnet.fc.in_features
        self.projection_head = SimCLRProjectionHead(hidden_dim, hidden_dim, 128)
        #print(hidden_dim)
        self.criterion = NTXentLoss(gather_distributed=True)

    def forward(self, x):
        h = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(h)
        return z

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log("train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(), lr=6e-2, momentum=0.9, weight_decay=0.00001
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, max_epochs
        )
        return [optim], [scheduler]