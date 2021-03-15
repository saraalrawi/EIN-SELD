import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    
    def calculate_attention(self, pred , pred_constraint, target, epoch_it,model):

        if 'PIT' not in self.cfg['training']['PIT_type']:
            updated_target = target
            loss_sed = self.losses[0].calculate_loss(pred['sed'], updated_target['sed'])
            loss_doa = self.losses[1].calculate_loss(pred['doa'], updated_target['doa'])
        elif self.cfg['training']['PIT_type'] == 'tPIT':
            loss_sed, loss_doa, updated_target, loss_doa_smoothness = self.tPIT(pred, target)
            if self.cfg['training']['weight_constraints'] and self.cfg['training']['model'] == 'EINV2':
                loss_orthogonal = self.orthogonal_distance(model)
            # stronger weight orthogonality, model EINV2
            if self.cfg['training']['weight_constraints_1'] and self.cfg['training']['model'] == 'EINV2':
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
            # stronger weight orthogonality, model SELD_ATT
            if self.cfg['training']['weight_constraints_1'] and self.cfg['training']['model'] == 'SELD_ATT':
                # orthogonal constraint on the conv layers of the shared_feature space
                loss_orthogonal = self.orth_dist(model.module.shared_conv_block1[0].weight)\
                                  + self.orth_dist(model.module.shared_conv_block1[3].weight)\
                                  + self.orth_dist(model.module.shared_conv_block2[0].weight)\
                                  + self.orth_dist(model.module.shared_conv_block2[3].weight) \
                                  + self.orth_dist(model.module.shared_conv_block3[0].weight) \
                                  + self.orth_dist(model.module.shared_conv_block3[3].weight) \
                                  + self.orth_dist(model.module.shared_conv_block4[0].weight) \
                                  + self.orth_dist(model.module.shared_conv_block4[3].weight)
                loss_orthogonal += self.deconv_orth_dist(model.module.shared_conv_block1[0].weight) \
                                   + self.deconv_orth_dist(model.module.shared_conv_block1[3].weight) \
                                   + self.deconv_orth_dist(model.module.shared_conv_block2[0].weight) \
                                   + self.deconv_orth_dist(model.module.shared_conv_block2[3].weight) \
                                   + self.deconv_orth_dist(model.module.shared_conv_block3[0].weight) \
                                   + self.deconv_orth_dist(model.module.shared_conv_block3[3].weight) \
                                   + self.deconv_orth_dist(model.module.shared_conv_block4[0].weight) \
                                   + self.deconv_orth_dist(model.module.shared_conv_block4[3].weight)
                # apply the constraint on the private spaces
                for j in range(len(model.module.encoder_att)):  # 2
                    for i in range(len(model.module.encoder_att[j])):  # 4
                        # apply orthogonality on 0 and 3 index of each block
                        loss_orthogonal += self.orth_dist(model.module.encoder_att[j][i][0].weight) \
                                           + self.orth_dist(model.module.encoder_att[j][i][3].weight)\
                                           + self.orth_dist(model.module.encoder_block_att[i][0].weight)\
                                           + self.deconv_orth_dist(model.module.encoder_att[j][i][0].weight)\
                                           + self.deconv_orth_dist(model.module.encoder_att[j][i][3].weight)\
                                           + self.deconv_orth_dist(model.module.encoder_block_att[i][0].weight)
            # apply constraint between the private spaces, only between the last conv layers of the attention modules
            if self.cfg['training']['layer_constraints_1'] and self.cfg['training']['model'] == 'SELD_ATT':
                loss_orthogonal = self.diff_loss(pred_constraint['sed_1'],pred_constraint['doa_1'])
            # apply constraint between the private spaces, only between the last conv layers of the attention modules
            if self.cfg['training']['layer_constraints_1'] and self.cfg['training']['model'] == 'SELD_ATT_LIGHT':
                loss_orthogonal = self.diff_loss(pred_constraint['sed_1'], pred_constraint['doa_1'])
            # orthogonality between the sed and doa branches of EINV2 model.
            if self.cfg['training']['layer_constraints_1'] and self.cfg['training']['model'] == 'EINV2':
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

        if self.cfg['training']['weight_constraints']:
            orthogonal_constraint_loss = self.adjust_ortho_decay_rate(epoch_it + 1) * loss_orthogonal
            loss_all = self.beta * loss_sed + (1 - self.beta) * loss_doa + orthogonal_constraint_loss

            losses_dict = {
                'all': loss_all,
                'sed': loss_sed,
                'doa': loss_doa,
                'loss_weight_orthogonal': orthogonal_constraint_loss,
                'updated_target': updated_target
                }
        elif self.cfg['training']['layer_constraints_1']:
                r = self.cfg['training']['r']
                orthogonal_constraint_loss = r * loss_orthogonal
                loss_all = self.beta * loss_sed + (1 - self.beta) * loss_doa + orthogonal_constraint_loss

                losses_dict = {
                    'all': loss_all,
                    'sed': loss_sed,
                    'doa': loss_doa,
                    'loss_layer_orthogonal_1': orthogonal_constraint_loss,
                    'updated_target': updated_target
                    }
        elif self.cfg['training']['weight_constraints_1']:
            # no weight decay self.cfg['training']['r']
            # self.args.r
            #r = self.cfg['training']['r']
            # EINV2-best=1e-5, SELD-ATTN=1e-3
            # new EINV2 dev folds 1e-3
            orthogonal_constraint_loss =  1e-3 * loss_orthogonal
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


    def calculate(self, pred , target, epoch_it,model):

        if 'PIT' not in self.cfg['training']['PIT_type']:
            updated_target = target
            loss_sed = self.losses[0].calculate_loss(pred['sed'], updated_target['sed'])
            loss_doa = self.losses[1].calculate_loss(pred['doa'], updated_target['doa'])
        elif self.cfg['training']['PIT_type'] == 'tPIT':
            loss_sed, loss_doa, updated_target, loss_doa_smoothness = self.tPIT(pred, target)
            if self.cfg['training']['weight_constraints'] and self.cfg['training']['model'] == 'EINV2':
                loss_orthogonal = self.orthogonal_distance(model)
            # stronger weight orthogonality, model EINV2
            if self.cfg['training']['weight_constraints_1'] and self.cfg['training']['model'] == 'EINV2':
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
            # stronger weight orthogonality, model SELD_ATT
            if self.cfg['training']['weight_constraints_1'] and self.cfg['training']['model'] == 'SELD_ATT':
                # orthogonal constraint on the conv layers of the shared_feature space
                loss_orthogonal = self.orth_dist(model.module.shared_conv_block1[0].weight)\
                                  + self.orth_dist(model.module.shared_conv_block1[3].weight)\
                                  + self.orth_dist(model.module.shared_conv_block2[0].weight)\
                                  + self.orth_dist(model.module.shared_conv_block2[3].weight) \
                                  + self.orth_dist(model.module.shared_conv_block3[0].weight) \
                                  + self.orth_dist(model.module.shared_conv_block3[3].weight) \
                                  + self.orth_dist(model.module.shared_conv_block4[0].weight) \
                                  + self.orth_dist(model.module.shared_conv_block4[3].weight)
                loss_orthogonal += self.deconv_orth_dist(model.module.shared_conv_block1[0].weight) \
                                   + self.deconv_orth_dist(model.module.shared_conv_block1[3].weight) \
                                   + self.deconv_orth_dist(model.module.shared_conv_block2[0].weight) \
                                   + self.deconv_orth_dist(model.module.shared_conv_block2[3].weight) \
                                   + self.deconv_orth_dist(model.module.shared_conv_block3[0].weight) \
                                   + self.deconv_orth_dist(model.module.shared_conv_block3[3].weight) \
                                   + self.deconv_orth_dist(model.module.shared_conv_block4[0].weight) \
                                   + self.deconv_orth_dist(model.module.shared_conv_block4[3].weight)
                # apply the constraint on the private spaces
                for j in range(len(model.module.encoder_att)):  # 2
                    for i in range(len(model.module.encoder_att[j])):  # 4
                        # apply orthogonality on 0 and 3 index of each block
                        loss_orthogonal += self.orth_dist(model.module.encoder_att[j][i][0].weight) \
                                           + self.orth_dist(model.module.encoder_att[j][i][3].weight)\
                                           + self.orth_dist(model.module.encoder_block_att[i][0].weight)\
                                           + self.deconv_orth_dist(model.module.encoder_att[j][i][0].weight)\
                                           + self.deconv_orth_dist(model.module.encoder_att[j][i][3].weight)\
                                           + self.deconv_orth_dist(model.module.encoder_block_att[i][0].weight)
            # apply constraint between the private spaces, only between the last conv layers of the attention modules
            #if self.cfg['training']['layer_constraints_1'] and self.cfg['training']['model'] == 'SELD_ATT':
            #    loss_orthogonal = self.diff_loss(pred_constraint['sed_1'],pred_constraint['doa_1'])
            # orthogonality between the sed and doa branches of EINV2 model.
            if self.cfg['training']['layer_constraints_1'] and self.cfg['training']['model'] == 'EINV2':
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

        if self.cfg['training']['weight_constraints']:
            orthogonal_constraint_loss = self.adjust_ortho_decay_rate(epoch_it + 1) * loss_orthogonal
            loss_all = self.beta * loss_sed + (1 - self.beta) * loss_doa + orthogonal_constraint_loss

            losses_dict = {
                'all': loss_all,
                'sed': loss_sed,
                'doa': loss_doa,
                'loss_weight_orthogonal': orthogonal_constraint_loss,
                'updated_target': updated_target
                }
        elif self.cfg['training']['layer_constraints_1']:
                r = self.cfg['training']['r']
                orthogonal_constraint_loss = r * loss_orthogonal
                loss_all = self.beta * loss_sed + (1 - self.beta) * loss_doa + orthogonal_constraint_loss

                losses_dict = {
                    'all': loss_all,
                    'sed': loss_sed,
                    'doa': loss_doa,
                    'loss_layer_orthogonal_1': orthogonal_constraint_loss,
                    'updated_target': updated_target
                    }
        elif self.cfg['training']['weight_constraints_1']:
            # no weight decay self.cfg['training']['r']
            # self.args.r
            #r = self.cfg['training']['r']
            orthogonal_constraint_loss =  1e-3 * loss_orthogonal
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
        '''
        This function implements weight orthogonality on the weights of all layers of the network.
        Args:
            model: e.g. EINV2

        Returns: unweighted orthogoanl loss.

        '''
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
                ident = Variable(torch.eye(cols, cols).cud())
                #ident = ident.cuda()

                w_tmp = (m - ident)
                height = w_tmp.size(0)
                u = F.normalize(w_tmp.new_empty(height).normal_(0, 1), dim=0, eps=1e-12)
                v = F.normalize(torch.matmul(w_tmp.t(), u), dim=0, eps=1e-12)
                u = F.normalize(torch.matmul(w_tmp, v), dim=0, eps=1e-12)
                sigma = torch.dot(u, torch.matmul(w_tmp, v))

                if l2_reg is None:
                    l2_reg = (sigma) ** 2
                else:
                    l2_reg = l2_reg + (sigma) ** 2
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

    def orth_dist(self, W, stride=None):
        '''
            This function is implemeted to impose orthogonality on the weight of conv layer.
                Args:
                    W: weight of a conv layer

                Returns: unweighted orthogoanl loss.

        '''
        cols = W[0].numel()
        rows = W.shape[0]
        w1 = W.view(-1, cols)
        wt = torch.transpose(w1, 0, 1)
        m = torch.matmul(wt, w1)
        ident = Variable(torch.eye(cols, cols).cuda())
        #ident = ident.cuda()

        w_tmp = (m - ident)
        height = w_tmp.size(0)
        u = F.normalize(w_tmp.new_empty(height).normal_(0, 1), dim=0, eps=1e-12)
        v = F.normalize(torch.matmul(w_tmp.t(), u), dim=0, eps=1e-12)
        u = F.normalize(torch.matmul(w_tmp, v), dim=0, eps=1e-12)
        sigma = torch.dot(u, torch.matmul(w_tmp, v))


        l2_reg = (sigma) ** 2

        return l2_reg

    # paper: https://arxiv.org/abs/1911.12207
    # For layers orthogonality
    def orth_dist_layer(self, W_1, W_2 ,stride=None):
        '''
        This function is implemented to impose orthogonality between sed and doa branches of EINV2 (baseline network)
        Args:
            W_1: weight from sed conv layer
            W_2: weight from doa conv layer
            stride:

        Returns: unweighted orthogoanl loss.

        '''
        cols = W_1[0].numel()
        rows = W_1.shape[0]
        w1 = W_2.view(-1, cols)
        wt = torch.transpose(w1, 0, 1)
        m = torch.matmul(wt, w1)
        ident = Variable(torch.eye(cols, cols).cuda())
        #ident = ident.cuda()

        w_tmp = (m - ident)
        height = w_tmp.size(0)
        u = F.normalize(w_tmp.new_empty(height).normal_(0, 1), dim=0, eps=1e-12)
        v = F.normalize(torch.matmul(w_tmp.t(), u), dim=0, eps=1e-12)
        u = F.normalize(torch.matmul(w_tmp, v), dim=0, eps=1e-12)
        sigma = torch.dot(u, torch.matmul(w_tmp, v))


        l2_reg = (sigma) ** 2

        return l2_reg
    # Stronger orthogonality from paper: https://arxiv.org/abs/1911.12207
    def deconv_orth_dist_layer(self,kernel,kernel_, stride=2, padding=1):
        [o_c, i_c, w, h] = kernel.shape
        output = torch.conv2d(kernel, kernel_, stride=stride, padding=padding)
        target = torch.zeros((o_c, o_c, output.shape[-2], output.shape[-1])).cuda()
        ct = int(np.floor(output.shape[-1] / 2))
        target[:, :, ct, ct] = torch.eye(o_c).cuda()
        return torch.norm(output - target)

    def diff_loss(self, input1, input2):
        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        input1_l2_norm = F.normalize(input1, p=2, dim=1).detach()
        #input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6) # should understand this

        input2_l2_norm = F.normalize(input2, p=2, dim=1).detach()
        #input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)  # should understand this

        diff_loss = torch.mean((input1_l2_norm.t().mm(input2_l2_norm)).pow(2)).cuda()
        return diff_loss

# diff loss from Domain Separation Networks.
class DiffLoss(nn.Module):

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):

        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss