import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch import nn
import numpy as np

from models.mlp import ProjectionMLP

class MM_HNL(LightningModule):
    """
    Multimodal adaptation of NTXent, according to the original CMC paper.
    """
    def __init__(self, batch_size, modalities, temperature=0.1, tau_plus=0.1, beta=1.0, estimator='hard'):
        super().__init__()
        self.batch_size = batch_size
        self.modalities = modalities
        self.temperature = temperature
        self.tau_plus = tau_plus
        self.beta = beta
        self.estimator = estimator

    def forward(self, x):
        loss, pos, neg = self.get_CMCHNL_loss(x, self.batch_size, self.modalities, self.temperature, self.tau_plus, self.beta, self.estimator)
        return loss, pos, neg
    
    @staticmethod
    def get_cosine_sim_matrix(features_1, features_2, temperature):
        features_1 = F.normalize(features_1, dim=1)
        features_2 = F.normalize(features_2, dim=1)
        #!!!look more into this if exp and temp is necessary
        similarity_matrix = torch.exp(torch.matmul(features_1, features_2.T.contiguous()) / temperature)
        return similarity_matrix

    def get_negative_mask(batch_size):
        negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
        for i in range(batch_size):
            negative_mask[i, i] = 0
            negative_mask[i, i + batch_size] = 0

        negative_mask = torch.cat((negative_mask, negative_mask), 0)
        return negative_mask

    def get_CMCHNL_loss(self, features, batch_size, modalities=2, temperature=0.1, tau_plus=0.1, beta=1.0, estimator='hard'):
        # Let M1 and M2 be abbreviations for the first and the second modality, respectively.
        
        # Computes similarity matrix by multiplication, shape: (batch_size, batch_size).
        # This computes the similarity between each sample in M1 with each sample in M2.
        #features_1 = F.normalize(features[modalities[0]], dim=1)
        #features_2 = F.normalize(features[modalities[1]], dim=1)
        features_1 = features[modalities[0]]
        features_2 = features[modalities[1]]
        '''
        # neg score
        out = torch.cat([features_1, features_2], dim=0)
        neg = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)

        #print("original", neg, neg.shape)
        old_neg = neg.clone()
        mask = MM_HNL.get_negative_mask(batch_size).cuda()
        neg = neg.masked_select(mask).view(2 * batch_size, -1)

        # pos score
        pos = torch.exp(torch.sum(features_1 * features_2, dim=-1) / temperature)
        pos = torch.cat([pos, pos], dim=0)
        '''
        similarity_matrix = MM_HNL.get_cosine_sim_matrix(features_1, features_2, temperature)
        
        # We need to formulate (2 * batch_size) instance discrimination problems:
        # -> each instance from M1 with each instance from M2
        # -> each instance from M2 with each instance from M1

        # Similarities on the main diagonal are from positive pairs, and are the same in both directions.
        mask = torch.eye(batch_size, dtype=torch.bool)
        positives_m1_m2 = similarity_matrix[mask].view(batch_size, -1)
        positives_m2_m1 = similarity_matrix[mask].view(batch_size, -1)
        pos = torch.cat([positives_m1_m2, positives_m2_m1], dim=0)

        # The rest of the similarities are from negative pairs. Row-wise for the loss from M1 to M2, and column-wise for the loss from M2 to M1.
        negatives_m1_m2 = similarity_matrix[~mask].view(batch_size, -1)
        negatives_m2_m1 = similarity_matrix.T[~mask].view(batch_size, -1)
        neg = torch.cat([negatives_m1_m2, negatives_m2_m1])
        
        # calculate the hard negative loss
        if estimator=='hard':
            N = batch_size * 2 - 2
            imp = (beta* neg.log()).exp()
            reweight_neg = (imp*neg).sum(dim = -1) / imp.mean(dim = -1)
            Ng = (-tau_plus * N * pos + reweight_neg) / (1 - tau_plus)
            # constrain (optional)
            Ng = torch.clamp(Ng, min = N * np.e**(-1 / temperature))
        elif estimator=='easy':
            Ng = neg.sum(dim=-1)
        else:
            raise Exception('Invalid estimator selected. Please use any of [hard, easy]')

        # contrastive loss
        loss = (- torch.log(pos / (pos + Ng) )).mean()

        return loss, pos.mean(), neg.mean()

class ContrastiveMultiviewCodingHNL(LightningModule):
    """
    Implementation of CMC (contrastive multiview coding), currently supporting exactly 2 views.
    """
    def __init__(self, modalities, encoders, temperature=0.1, tau_plus=0.1, beta=1.0, estimator='easy', lr=0.001, weight_decay = 0, batch_size=64, hidden=[256, 128], optimizer_name_ssl='adam', **kwargs):
        super().__init__()
        self.save_hyperparameters('modalities', 'hidden', 'batch_size', 'temperature', 'optimizer_name_ssl', 'lr')
        
        self.modalities = modalities
        self.encoders = nn.ModuleDict(encoders)

        self.projections = {}
        for m in modalities:
            self.projections[m] = ProjectionMLP(in_size=encoders[m].out_size, hidden=hidden)
        self.projections = nn.ModuleDict(self.projections)

        self.optimizer_name_ssl = optimizer_name_ssl
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss = MM_HNL(batch_size, modalities, temperature, tau_plus, beta, estimator)

    def _forward_one_modality(self, modality, inputs):
        x = inputs[modality]
        x = self.encoders[modality](x)
        x = nn.Flatten()(x)
        x = self.projections[modality](x)
        return x

    def forward(self, x):
        outs = {}
        for m in self.modalities:
            outs[m] = self._forward_one_modality(m, x)
        return outs

    def training_step(self, batch, batch_idx):
        for m in self.modalities:
            batch[m] = batch[m].float()
        outs = self(batch)
        loss, pos, neg = self.loss(outs)
        self.log("ssl_train_loss", loss)
        self.log("avg_positive_sim", pos)
        self.log("avg_neg_sim", neg)
        return loss

    def validation_step(self, batch, batch_idx):
        for m in self.modalities:
            batch[m] = batch[m].float()
        outs = self(batch)
        loss, _, _ = self.loss(outs)
        self.log("ssl_val_loss", loss)

    def configure_optimizers(self):
        return self._initialize_optimizer()

    def _initialize_optimizer(self):
        if self.optimizer_name_ssl.lower() == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": 'ssl_train_loss'
                }
            }