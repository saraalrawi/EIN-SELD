import torch
import torch.nn as nn
import torch.nn.functional as F
eps = torch.finfo(torch.float32).eps



class WeightsOrthogonalityConstraint(object):
    def __init__(self,weight):
        self.weight = weight

    def __call__(self, *args, **kwargs):
        pass