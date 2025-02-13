import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl


from PIL import Image
import numpy as np
from lightly.models.modules.heads import SimCLRProjectionHead
from lightly.loss import NTXentLoss

from stainlib.augmentation.augmenter import HedColorAugmenter1


class HedColorAug:
    def __init__(self, hed_thresh = 0.03):
        self.hed_thresh = hed_thresh
    def __call__(self, image):
        #print(type(image))
        dab_lighter_aug = HedColorAugmenter1(self.hed_thresh)
        dab_lighter_aug.randomize()
        return Image.fromarray(dab_lighter_aug.transform(np.array(image)))

path_to_model = 'tenpercent_resnet18.ckpt'
max_epochs = 25

def load_model_weights(model, weights):
    model_dict = model.state_dict()
    weights = {k: v for k, v in weights.items() if k in model_dict}
    if weights == {}:
        print('No weight could be loaded..')
    model_dict.update(weights)
    model.load_state_dict(model_dict)
    return model

class SimCLRModel(pl.LightningModule):
    def __init__(self, lambda_ssl = 0.90):
        super().__init__()

        self.lambda_ssl = lambda_ssl
        resnet = torchvision.models.__dict__['resnet18'](pretrained=False)
        state = torch.load(path_to_model)
        state_dict = state['state_dict']
        for key in list(state_dict.keys()):
            state_dict[key.replace('model.', '').replace('resnet.', '')] = state_dict.pop(key)
        resnet = load_model_weights(resnet, state_dict)

        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        #print(self.backbone)
        hidden_dim = resnet.fc.in_features
        self.projection_head = SimCLRProjectionHead(hidden_dim, hidden_dim, 128)
        self.regression_head = nn.Linear(hidden_dim, 1)
        #print(hidden_dim)
        # contrastive learning
        self.criterion_ssl = NTXentLoss(gather_distributed=True)
        #Regression
        self.criterion_reg = nn.MSELoss()


    def forward(self, x):
        h = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(h)
        y = self.regression_head(h)
        return h, z, y

    def training_step(self, batch, batch_idx):

        images, labels = batch
        total_loss = 0

        labeled_idx = labels != -1
        #unlabeled_idx = labels == -1


        # #print(labeled_idx, labeled_idx.shape)


        if labeled_idx.any():
            labeled_images = images[0][labeled_idx]
            #print(labeled_images.shape)
            labeled_labels = labels[labeled_idx].float().unsqueeze(1)

            _, _, y_pred = self.forward(labeled_images)
            loss_reg = self.criterion_reg(y_pred, labeled_labels)
            total_loss += (1 - self.lambda_ssl) * loss_reg
            self.log("train_loss_reg", loss_reg)

        view1 = images[0]
        view2 = images[1]
        # view1, view2 = unlabeled_images[:, 0], unlabeled_images[:, 1]
        _, z1, _ = self.forward(view1)
        _, z2, _ = self.forward(view2)

        loss_ssl = self.criterion_ssl(z1, z2)
        total_loss += self.lambda_ssl * loss_ssl
        self.log("train_loss_ssl", loss_ssl)


        self.log("train_total_loss", total_loss)
        return total_loss

    def configure_optimizers(self):
        #SGD ADAM RMSPROP
        optim = torch.optim.Adam(
            self.parameters(),
            lr=1e-3,
            #momentum=0.9,
            weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, max_epochs
        )
        return [optim], [scheduler]