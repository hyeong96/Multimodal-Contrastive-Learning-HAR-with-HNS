import torch
import torch.nn.functional as F
#from pl_bolts.optimizers.lars import LARS
#from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay
from pytorch_lightning import LightningModule
from torch import nn
import numpy as np

from models.mlp import ProjectionMLP


class HardNegativeLoss(LightningModule):
    def __init__(self, batch_size, n_views=2, temperature=0.1, tau_plus=0.1, beta=1.0, estimator='hard'):
        super().__init__()
        self.batch_size = batch_size
        self.n_views = n_views
        self.temperature = temperature
        self.tau_plus = tau_plus
        self.beta = beta
        self.estimator = estimator

    def forward(self, x):
        loss, labels, pos, neg = self.get_HardNegative_loss(x, self.batch_size, self.n_views, self.temperature, self.tau_plus, self.beta, self.estimator)
        return loss, pos, neg
    
    def get_HardNegative_loss(self, features, batch_size, n_views=2, temperature=0.1, tau_plus=0.1, beta=1.0, estimator='hard'):
        """
            Implementation from https://github.com/sthalles/SimCLR/blob/master/simclr.py
        """
        # creates a vector with labels [0, 1, 2, 0, 1, 2] 
        labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
        # creates matrix where 1 is on the main diagonal and where indexes of the same intances match (e.g. [0, 4][1, 5] for batch_size=3 and n_views=2) 
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        # computes similarity matrix by multiplication, shape: (batch_size * n_views, batch_size * n_views)
        similarity_matrix = get_cosine_sim_matrix(features, temperature)
        
        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool)#.to(self.args.device)
        # mask out the main diagonal - output has one column less 
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix_wo_diag = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        
        # select and combine multiple positives
        pos = similarity_matrix_wo_diag[labels.bool()].view(labels.shape[0], -1)
        # select only the negatives 
        neg = similarity_matrix_wo_diag[~labels.bool()].view(similarity_matrix_wo_diag.shape[0], -1)
        
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



        return loss, labels.long().to(loss.device), pos.mean(), neg.mean()


def get_cosine_sim_matrix(features, temperature):
    #features = F.normalize(features, dim=1)
    #print(features)
    similarity_matrix = torch.exp(torch.matmul(features, features.T.contiguous()) / temperature)
    return similarity_matrix


class SimCLRUnimodal(LightningModule):
    def __init__(self, modality, encoder, mlp_in_size, temperature=0.1, tau_plus=0.1, beta=1.0, estimator='easy', batch_size=32, lr=0.001, momentum=0.9, weight_decay=1e-4, hidden=[256, 128], n_views=2, optimizer_name_ssl='lars', **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters('modality', 'hidden', 'batch_size', 'temperature', 'tau_plus', 'beta', 'estimator', 'n_views', 'optimizer_name_ssl', 'lr')
        self.encoder = encoder
        self.projection = ProjectionMLP(mlp_in_size, hidden)
        self.modality = modality

        self.optimizer_name_ssl = optimizer_name_ssl
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.loss = HardNegativeLoss(batch_size, n_views, temperature, tau_plus, beta, estimator)

    def forward(self, x):
        x = self.encoder(x)
        x = nn.Flatten()(x)
        x = self.projection(x)
        return x

    def training_step(self, batch, batch_idx):
        batch = torch.cat(batch[self.modality], dim=0).float()
        out = self(batch)
        loss, pos, neg = self.loss(out)
        self.log("ssl_train_loss", loss)
        self.log("avg_positive_sim", pos)
        self.log("avg_neg_sim", neg)
        return loss

    def validation_step(self, batch, batch_idx):
        batch = torch.cat(batch[self.modality], dim=0).float()
        out = self(batch)
        loss, _, _ = self.loss(out)
        self.log("ssl_val_loss", loss)

    def configure_optimizers(self):
        return self._initialize_optimizer()

    def _initialize_optimizer(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": 'ssl_train_loss'
            }
        }
        """
        if self.optimizer_name_ssl.lower() == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": 'ssl_train_loss'
                }
            }

        elif self.optimizer_name_ssl.lower() == 'lars':
            optimizer = LARS(
                self.parameters(),
                self.lr,
                momentum=0.9,
                weight_decay=1e-6,
                trust_coefficient=0.001
            )
            return {
                "optimizer": optimizer
            }
        """