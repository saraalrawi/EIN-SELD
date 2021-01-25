import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import normalize
from torch.autograd import Variable
from methods.utils.loss_utilities import BCEWithLogitsLoss, MSELoss


class Losses:
    def __init__(self, cfg, args):
        
        self.cfg = cfg
        self.args = args
        self.beta = cfg['training']['loss_beta']
        self.decay = cfg['training']['orthogonal_decay']
        self.losses = [BCEWithLogitsLoss(reduction='mean'), MSELoss(reduction='mean')]
        self.losses_pit = [BCEWithLogitsLoss(reduction='PIT'), MSELoss(reduction='PIT')]


        self.names = ['loss_all'] + [loss.name for loss in self.losses]
    
    def calculate(self, pred, target, epoch_it,model):

        if 'PIT' not in self.cfg['training']['PIT_type']:
            updated_target = target
            loss_sed = self.losses[0].calculate_loss(pred['sed'], updated_target['sed'])
            loss_doa = self.losses[1].calculate_loss(pred['doa'], updated_target['doa'])
        elif self.cfg['training']['PIT_type'] == 'tPIT':
            loss_sed, loss_doa, updated_target, loss_doa_smoothness = self.tPIT(pred, target)
            if self.cfg['training']['weight_constraints'] == 'orthogonal':
                loss_orthogonal = self.orthogonal_distance(model)

            # stronger weight orthogonality
            if self.cfg['training']['weight_constraints_1'] == 'orthogonal':
                loss_orthogonal = self.orth_dist(model.module.sed_conv_block1[0].double_conv[0].weight) \
                                  + self.orth_dist(model.module.sed_conv_block1[0].double_conv[3].weight) \
                                  + self.orth_dist(model.module.sed_conv_block2[0].double_conv[0].weight) \
                                  + self.orth_dist(model.module.sed_conv_block2[0].double_conv[3].weight) \
                                  + self.orth_dist(model.module.sed_conv_block3[0].double_conv[0].weight) \
                                  + self.orth_dist(model.module.sed_conv_block3[0].double_conv[3].weight) \
                                  + self.orth_dist(model.module.sed_conv_block4[0].double_conv[0].weight) \
                                  + self.orth_dist(model.module.sed_conv_block4[0].double_conv[3].weight) \
                                  + self.orth_dist(model.module.doa_conv_block1[0].double_conv[0].weight)  \
                                  + self.orth_dist(model.module.doa_conv_block1[0].double_conv[3].weight) \
                                  + self.orth_dist(model.module.doa_conv_block2[0].double_conv[0].weight)  \
                                  + self.orth_dist(model.module.doa_conv_block2[0].double_conv[3].weight)  \
                                  + self.orth_dist(model.module.doa_conv_block3[0].double_conv[0].weight) \
                                  + self.orth_dist(model.module.doa_conv_block3[0].double_conv[3].weight) \
                                  + self.orth_dist(model.module.doa_conv_block4[0].double_conv[0].weight) \
                                  + self.orth_dist(model.module.doa_conv_block4[0].double_conv[3].weight)

                loss_orthogonal += self.deconv_orth_dist(model.module.sed_conv_block1[0].double_conv[0].weight, stride=1) + self.deconv_orth_dist(
                    model.module.sed_conv_block1[0].double_conv[3].weight) + self.deconv_orth_dist(
                    model.module.sed_conv_block2[0].double_conv[0].weight)
                loss_orthogonal += self.deconv_orth_dist(model.module.sed_conv_block2[0].double_conv[3].weight, stride=1) + self.deconv_orth_dist(
                    model.module.sed_conv_block3[0].double_conv[0].weight, stride=1) + self.deconv_orth_dist(
                    model.module.sed_conv_block3[0].double_conv[3].weight, stride=1)
                loss_orthogonal += self.deconv_orth_dist(model.module.sed_conv_block4[0].double_conv[0].weight, stride=1) + self.deconv_orth_dist(
                    model.module.sed_conv_block4[0].double_conv[3].weight, stride=1) + self.deconv_orth_dist(
                    model.module.doa_conv_block1[0].double_conv[0].weight, stride=1) + self.deconv_orth_dist(
                    model.module.doa_conv_block1[0].double_conv[3].weight, stride=1) + self.deconv_orth_dist(
                    model.module.doa_conv_block2[0].double_conv[0].weight, stride=1) + self.deconv_orth_dist(
                    model.module.doa_conv_block2[0].double_conv[3].weight, stride=1)
                loss_orthogonal += self.deconv_orth_dist(model.module.doa_conv_block3[0].double_conv[0].weight, stride=1) + self.deconv_orth_dist(
                    model.module.doa_conv_block3[0].double_conv[3].weight, stride=1) + self.deconv_orth_dist(
                    model.module.doa_conv_block4[0].double_conv[0].weight, stride=1) + self.deconv_orth_dist(
                    model.module.doa_conv_block4[0].double_conv[3].weight, stride=1)

            if self.cfg['training']['layer_constraints_1'] == 'orthogonal':
                loss_orthogonal = self.orth_dist_layer(model.module.sed_conv_block1[0].double_conv[0].weight,model.module.doa_conv_block1[0].double_conv[0].weight) \
                                  + self.orth_dist_layer(model.module.sed_conv_block1[0].double_conv[3].weight, model.module.doa_conv_block1[0].double_conv[3].weight) \
                                  + self.orth_dist_layer(model.module.sed_conv_block2[0].double_conv[0].weight,model.module.doa_conv_block2[0].double_conv[0].weight) \
                                  + self.orth_dist_layer(model.module.sed_conv_block2[0].double_conv[3].weight,model.module.doa_conv_block2[0].double_conv[3].weight) \
                                  + self.orth_dist_layer(model.module.sed_conv_block3[0].double_conv[0].weight,model.module.doa_conv_block3[0].double_conv[0].weight) \
                                  + self.orth_dist_layer(model.module.sed_conv_block3[0].double_conv[3].weight,model.module.doa_conv_block3[0].double_conv[3].weight) \
                                  + self.orth_dist_layer(model.module.sed_conv_block4[0].double_conv[0].weight, model.module.doa_conv_block4[0].double_conv[0].weight) \
                                  + self.orth_dist_layer(model.module.sed_conv_block4[0].double_conv[3].weight,model.module.doa_conv_block4[0].double_conv[3].weight)

                loss_orthogonal += self.deconv_orth_dist_layer(model.module.sed_conv_block1[0].double_conv[0].weight,model.module.doa_conv_block1[0].double_conv[0].weight ,
                                                         stride=1) + self.deconv_orth_dist_layer(
                    model.module.sed_conv_block1[0].double_conv[3].weight,model.module.doa_conv_block1[0].double_conv[3].weight) + self.deconv_orth_dist_layer(
                    model.module.sed_conv_block2[0].double_conv[0].weight, model.module.doa_conv_block2[0].double_conv[0].weight)
                loss_orthogonal += self.deconv_orth_dist_layer(model.module.sed_conv_block2[0].double_conv[3].weight,model.module.doa_conv_block2[0].double_conv[3].weight ,
                                                         stride=1) + self.deconv_orth_dist_layer(
                    model.module.sed_conv_block3[0].double_conv[0].weight, model.module.doa_conv_block3[0].double_conv[0].weight,  stride=1) + self.deconv_orth_dist_layer(
                    model.module.sed_conv_block3[0].double_conv[3].weight,model.module.doa_conv_block3[0].double_conv[3].weight ,stride=1)
                loss_orthogonal += self.deconv_orth_dist_layer(model.module.sed_conv_block4[0].double_conv[0].weight,model.module.doa_conv_block4[0].double_conv[0].weight ,
                                                         stride=1) + self.deconv_orth_dist_layer(
                    model.module.sed_conv_block4[0].double_conv[3].weight,model.module.doa_conv_block4[0].double_conv[3].weight, stride=1)

        if self.cfg['training']['weight_constraints'] == 'orthogonal':
            orthogonal_constraint_loss = self.adjust_ortho_decay_rate(epoch_it + 1) * loss_orthogonal
            loss_all = self.beta * loss_sed + (1 - self.beta) * loss_doa + orthogonal_constraint_loss

            losses_dict = {
                'all': loss_all,
                'sed': loss_sed,
                'doa': loss_doa,
                'loss_weight_orthogonal': orthogonal_constraint_loss,
                'updated_target': updated_target
                }
        elif self.cfg['training']['layer_constraints_1'] == 'orthogonal':
                orthogonal_constraint_loss = self.args.r * loss_orthogonal
                loss_all = self.beta * loss_sed + (1 - self.beta) * loss_doa + orthogonal_constraint_loss

                losses_dict = {
                    'all': loss_all,
                    'sed': loss_sed,
                    'doa': loss_doa,
                    'loss_layer_orthogonal_1': orthogonal_constraint_loss,
                    'updated_target': updated_target
                    }
        elif self.cfg['training']['weight_constraints_1'] == 'orthogonal':
            # no weight decay self.cfg['training']['r']
            orthogonal_constraint_loss =  self.args.r * loss_orthogonal
            loss_all = self.beta * loss_sed + (1 - self.beta) * loss_doa + orthogonal_constraint_loss

            losses_dict = {
                'all': loss_all,
                'sed': loss_sed,
                'doa': loss_doa,
                'loss_weight_orthogonal_1': orthogonal_constraint_loss,
                'updated_target': updated_target
                }
        elif self.cfg['training']['smoothness_loss']:
            smoothness_weight = 1
            loss_all = self.beta * loss_sed + (1 - self.beta) * loss_doa + smoothness_weight * (loss_doa_smoothness)
            losses_dict = {
                'all': loss_all,
                'sed': loss_sed,
                'doa': loss_doa,
                'loss_doa_smoothness': loss_doa_smoothness,
                'updated_target': updated_target,
                }
        else:
            loss_all = self.beta * loss_sed + (1 - self.beta) * loss_doa
            losses_dict = {
                'all': loss_all,
                'sed': loss_sed,
                'doa': loss_doa,
                'updated_target': updated_target
            }
        return losses_dict



    def tPIT(self, pred, target):
        """Frame Permutation Invariant Training for 2 possible combinations

        Args:
            pred: {
                'sed': [batch_size, T, num_tracks=2, num_classes], 
                'doa': [batch_size, T, num_tracks=2, doas=3]
            }
            target: {
                'sed': [batch_size, T, num_tracks=2, num_classes], 
                'doa': [batch_size, T, num_tracks=2, doas=3]            
            }
        Return:
            updated_target: updated target with the minimum loss frame-wisely
                {
                    'sed': [batch_size, T, num_tracks=2, num_classes], 
                    'doa': [batch_size, T, num_tracks=2, doas=3]            
                }
        """
        target_flipped = {
            'sed': target['sed'].flip(dims=[2]),
            'doa': target['doa'].flip(dims=[2])
        }

        loss_sed1 = self.losses_pit[0].calculate_loss(pred['sed'], target['sed'])
        loss_sed2 = self.losses_pit[0].calculate_loss(pred['sed'], target_flipped['sed'])
        loss_doa1 = self.losses_pit[1].calculate_loss(pred['doa'], target['doa'])
        loss_doa2 = self.losses_pit[1].calculate_loss(pred['doa'], target_flipped['doa'])

        loss1 = loss_sed1 + loss_doa1
        loss2 = loss_sed2 + loss_doa2

        loss_sed = (loss_sed1 * (loss1 <= loss2) + loss_sed2 * (loss1 > loss2)).mean()
        loss_doa = (loss_doa1 * (loss1 <= loss2) + loss_doa2 * (loss1 > loss2)).mean()
        loss_doa_smoothness = 0

        if self.cfg['training']['smoothness_loss']:
            smoothness_threshold = 1.0
            doa_array = pred['doa']

            # 1st derivative
            d_doa_array_dt = doa_array[:, 1:, :, :] - doa_array[:, :-1, :, :]

            # 2nd derivative
            d2_doa_array_dt2 = d_doa_array_dt[:, 1:, :, :] - d_doa_array_dt[:, :-1, :, :]

            # ignore non-significant non-smoothness
            d2_doa_array_dt2 = torch.where(d2_doa_array_dt2 < smoothness_threshold,
                                           torch.zeros_like(d2_doa_array_dt2), d2_doa_array_dt2)

            # actual loss is MSE of d2_doa_array_dt2
            loss_doa_smoothness = (d2_doa_array_dt2 ** 2).mean()
            #loss_doa = loss_doa_smoothness * loss_doa


        updated_target_sed = target['sed'].clone() * (loss1[:, :, None, None] <= loss2[:, :, None, None]) + \
            target_flipped['sed'].clone() * (loss1[:, :, None, None] > loss2[:, :, None, None])
        updated_target_doa = target['doa'].clone() * (loss1[:, :, None, None] <= loss2[:, :, None, None]) + \
            target_flipped['doa'].clone() * (loss1[:, :, None, None] > loss2[:, :, None, None])
        updated_target = {
            'sed': updated_target_sed,
            'doa': updated_target_doa
        }
        return loss_sed, loss_doa, updated_target, loss_doa_smoothness

    def orthogonal_distance(self, model):
        l2_reg = None
        for W in model.parameters():
            if W.ndimension() < 2:
                continue
            else:
                cols = W[0].numel()
                rows = W.shape[0]
                w1 = W.view(-1, cols)
                wt = torch.transpose(w1, 0, 1)
                m = torch.matmul(wt, w1)
                ident = Variable(torch.eye(cols, cols))
                ident = ident.cuda()

                w_tmp = (m - ident)
                height = w_tmp.size(0)
                u = normalize(w_tmp.new_empty(height).normal_(0, 1), dim=0, eps=1e-12)
                v = normalize(torch.matmul(w_tmp.t(), u), dim=0, eps=1e-12)
                u = normalize(torch.matmul(w_tmp, v), dim=0, eps=1e-12)
                sigma = torch.dot(u, torch.matmul(w_tmp, v))

                if l2_reg is None:
                    l2_reg = (sigma) ** 2
                else:
                    l2_reg = l2_reg + (sigma) ** 2
        return l2_reg

    def orthogonal_layer_distance(self,sed_layer ,doa_layer):
        #l2_reg = None

        cols = sed_layer[0].numel()
        cols_doa = doa_layer[0].numel()
        rows = sed_layer.shape[0]
        w1 = sed_layer.view(-1, cols)
        w2 = doa_layer.view(-1, cols_doa)
        wt = torch.transpose(w2, 0, 1)
        m = torch.matmul(wt, w1)
        ident = Variable(torch.eye(cols, cols))
        ident = ident.cuda()

        w_tmp = (m - ident)
        height = w_tmp.size(0)
        u = normalize(w_tmp.new_empty(height).normal_(0, 1), dim=0, eps=1e-12)
        v = normalize(torch.matmul(w_tmp.t(), u), dim=0, eps=1e-12)
        u = normalize(torch.matmul(w_tmp, v), dim=0, eps=1e-12)
        sigma = torch.dot(u, torch.matmul(w_tmp, v))


        l2_reg = (sigma) ** 2

        return l2_reg

    def adjust_ortho_decay_rate(self,epoch_it):
        o_d = self.decay

        if epoch_it > 215:
            o_d = 0.0
        elif epoch_it > 100:
            o_d = 1e-6 * o_d
        elif epoch_it > 60:
            o_d = 1e-4 * o_d
        elif epoch_it > 40:
            o_d = 1e-3 * o_d

        return o_d

    # paper: https://arxiv.org/abs/1911.12207
    # For Kernel orthogonality
    def deconv_orth_dist(self,kernel, stride=2, padding=1):
        [o_c, i_c, w, h] = kernel.shape
        output = torch.conv2d(kernel, kernel, stride=stride, padding=padding)
        target = torch.zeros((o_c, o_c, output.shape[-2], output.shape[-1])).cuda()
        ct = int(np.floor(output.shape[-1] / 2))
        target[:, :, ct, ct] = torch.eye(o_c).cuda()
        return torch.norm(output - target)

    '''
    def orth_dist(self,matr, stride=None):
        mat = matr.reshape((matr.shape[0], -1))
        if matr.shape[0] < matr.shape[1]:
            matr = matr.permute(1, 0)
        return torch.norm(torch.t(matr) @ mat - torch.eye(matr.shape[1]).cuda())
    '''
    def orth_dist(self, W, stride=None):
        cols = W[0].numel()
        rows = W.shape[0]
        w1 = W.view(-1, cols)
        wt = torch.transpose(w1, 0, 1)
        m = torch.matmul(wt, w1)
        ident = Variable(torch.eye(cols, cols))
        ident = ident.cuda()

        w_tmp = (m - ident)
        height = w_tmp.size(0)
        u = normalize(w_tmp.new_empty(height).normal_(0, 1), dim=0, eps=1e-12)
        v = normalize(torch.matmul(w_tmp.t(), u), dim=0, eps=1e-12)
        u = normalize(torch.matmul(w_tmp, v), dim=0, eps=1e-12)
        sigma = torch.dot(u, torch.matmul(w_tmp, v))


        l2_reg = (sigma) ** 2

        return l2_reg

    # paper: https://arxiv.org/abs/1911.12207
    # For layers orthogonality
    def orth_dist_layer(self, W_1, W_2 ,stride=None):
        cols = W_1[0].numel()
        rows = W_1.shape[0]
        w1 = W_2.view(-1, cols)
        wt = torch.transpose(w1, 0, 1)
        m = torch.matmul(wt, w1)
        ident = Variable(torch.eye(cols, cols))
        ident = ident.cuda()

        w_tmp = (m - ident)
        height = w_tmp.size(0)
        u = normalize(w_tmp.new_empty(height).normal_(0, 1), dim=0, eps=1e-12)
        v = normalize(torch.matmul(w_tmp.t(), u), dim=0, eps=1e-12)
        u = normalize(torch.matmul(w_tmp, v), dim=0, eps=1e-12)
        sigma = torch.dot(u, torch.matmul(w_tmp, v))


        l2_reg = (sigma) ** 2

        return l2_reg

    def deconv_orth_dist_layer(self,kernel,kernel_, stride=2, padding=1):
        [o_c, i_c, w, h] = kernel.shape
        output = torch.conv2d(kernel, kernel_, stride=stride, padding=padding)
        target = torch.zeros((o_c, o_c, output.shape[-2], output.shape[-1])).cuda()
        ct = int(np.floor(output.shape[-1] / 2))
        target[:, :, ct, ct] = torch.eye(o_c).cuda()
        return torch.norm(output - target)