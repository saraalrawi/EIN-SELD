import torch
import torch.nn as nn
import torch.nn.functional as F
eps = torch.finfo(torch.float32).eps


class MSELoss:
    def __init__(self, reduction='mean'):
        self.reduction = reduction
        self.name = 'loss_MSE'
        if self.reduction != 'PIT':
            self.loss = nn.MSELoss(reduction='mean')
        else:
            self.loss = nn.MSELoss(reduction='none')
    
    def calculate_loss(self, pred, target):
        if self.reduction != 'PIT':
            return self.loss(pred, target)
        else:
            return self.loss(pred, target).mean(dim=tuple(range(2, pred.ndim)))


class BCEWithLogitsLoss:
    def __init__(self, reduction='mean', pos_weight=None):
        self.reduction = reduction
        self.name = 'loss_BCEWithLogits'
        if self.reduction != 'PIT':
            self.loss = nn.BCEWithLogitsLoss(reduction=self.reduction, pos_weight=pos_weight)
        else:
            self.loss = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)
    
    def calculate_loss(self, pred, target):
        if self.reduction != 'PIT':
            return self.loss(pred, target)
        else:
            return self.loss(pred, target).mean(dim=tuple(range(2, pred.ndim)))

class OrthogonalLoss(nn.Module):
    def __init__(self):
        super(OrthogonalLoss, self).__init__()
    def forward(self, input1, input2):
        diff_loss = 0
        return diff_loss

    def calculate_orthogonal_loss(mat_1, mat_2, stride=None):
        """
        this function finds the orthogonality distance between the layers in the model.
        Params:
            mat_1: layer 1
            mat_2: layer 2
        Returns:
            orth_dist: orthogonality distance
        """
        mat_1 = mat_1.reshape((mat_1.shape[0], -1))
        if mat_1.shape[0] < mat_1.shape[1]:
            mat = mat_1.permute(1, 0)
        orth_dist = torch.norm(torch.t(mat_1) @ mat_2 - torch.eye(mat_1.shape[1]).cuda())
        return orth_dist