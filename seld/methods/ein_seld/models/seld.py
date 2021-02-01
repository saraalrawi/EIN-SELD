import torch
import torch.nn as nn
import torch.nn.functional as F
from methods.utils.model_utilities import (DoubleConv, PositionalEncoding,
                                           init_layer, AttentionModule)

class SELD_ATT(nn.Module):
    def __init__(self, cfg, dataset):
        super().__init__()
        self.pe_enable = False  # Ture | False
        self.cfg = cfg

        filter = [64, 128, 256, 512, 512]

        if cfg['data']['audio_feature'] == 'logmel&intensity':
            self.f_bins = cfg['data']['n_mels']
            self.in_channels = 7

        # defining shared network
        self.shared_conv_block1 = nn.Sequential(nn.Conv2d(in_channels=self.in_channels,
                    out_channels=64,
                    kernel_size=(3, 3), stride=(1, 1),
                    padding=(1,1), dilation=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=64,
                    out_channels=64,
                    kernel_size=(3,3), stride=(1,1),
                    padding=(1,1), dilation=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),nn.AvgPool2d(kernel_size=(2, 2)))

        self.shared_conv_block2 = nn.Sequential(nn.Conv2d(in_channels=64,
                                                          out_channels=128,
                                                          kernel_size=(3, 3),
                                                          stride=(1, 1),
                                                          padding=(1,1),
                                                          dilation=1,
                                                          bias=False),
                                                nn.BatchNorm2d(128),
                                                nn.ReLU(inplace=True),
            # nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                            nn.Conv2d(in_channels=128,
                                                      out_channels=128,
                                                      kernel_size=(3,3),
                                                      stride=(1,1),
                                                      padding=(1,1),
                                                      dilation=1,
                                                      bias=False),
                                            nn.BatchNorm2d(128),
                                            nn.ReLU(inplace=True),
                                            nn.AvgPool2d(kernel_size=(2, 2)))

        self.shared_conv_block3 = nn.Sequential(nn.Conv2d(in_channels=128,
                                                          out_channels=256,
                                                          kernel_size=(3, 3), stride=(1, 1),
                                                          padding=(1, 1), dilation=1, bias=False),
                                                nn.BatchNorm2d(256),
                                                nn.ReLU(inplace=True),
                                                # nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                                nn.Conv2d(in_channels=256,
                                                          out_channels=256,
                                                          kernel_size=(3, 3), stride=(1, 1),
                                                          padding=(1, 1), dilation=1, bias=False),
                                                nn.BatchNorm2d(256),
                                                nn.ReLU(inplace=True),nn.AvgPool2d(kernel_size=(1, 2)) )
        self.shared_conv_block4 = nn.Sequential(nn.Conv2d(in_channels=256,
                                                          out_channels=512,
                                                          kernel_size=(3, 3),
                                                          stride=(1, 1),
                                                          padding=(1, 1),
                                                          dilation=1,
                                                          bias=False),
                                                nn.BatchNorm2d(512),
                                                nn.ReLU(inplace=True),
                                                # nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                                nn.Conv2d(in_channels=512,
                                                          out_channels=512,
                                                          kernel_size=(3, 3),
                                                          stride=(1, 1),
                                                          padding=(1, 1),
                                                          dilation=1,
                                                          bias=False),
                                                nn.BatchNorm2d(512),
                                                nn.ReLU(inplace=True),
                                                nn.AvgPool2d(kernel_size=(1, 2))) # nn.AvgPool2d(kernel_size=(1, 2))

        # define task attention layers
        self.encoder_att = nn.ModuleList([nn.ModuleList([self.att_layer([filter[0], filter[0], filter[0]])])])
        self.encoder_block_att = nn.ModuleList([self.conv_layer([filter[0], filter[1]])])
        self.shared_network = [self.shared_conv_block1, self.shared_conv_block2, self.shared_conv_block3, self.shared_conv_block4]

        for j in range(2):
            if j < 1:
                self.encoder_att.append(nn.ModuleList([self.att_layer([filter[0], filter[0], filter[0]])]))
            for i in range(3):
                self.encoder_att[j].append(self.att_layer([2 * filter[i + 1], filter[i + 1], filter[i + 1]]))

        for i in range(3):
            if i < 3:
                self.encoder_block_att.append(self.conv_layer([filter[i + 1], filter[i + 2]]))
            else:
                self.encoder_block_att.append(self.conv_layer([filter[i + 1], filter[i + 1]]))


        if self.pe_enable:
            self.pe = PositionalEncoding(pos_len=100, d_model=512, pe_type='t', dropout=0.0)
        self.sed_trans_track1 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=1024, dropout=0.2), num_layers=2)
        self.sed_trans_track2 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=1024, dropout=0.2), num_layers=2)
        self.doa_trans_track1 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=1024, dropout=0.2), num_layers=2)
        self.doa_trans_track2 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=1024, dropout=0.2), num_layers=2)

        self.fc_sed_track1 = nn.Linear(512, 14, bias=True)
        self.fc_sed_track2 = nn.Linear(512, 14, bias=True)
        self.fc_doa_track1 = nn.Linear(512, 3, bias=True)
        self.fc_doa_track2 = nn.Linear(512, 3, bias=True)
        self.final_act_sed = nn.Sequential()  # nn.Sigmoid()
        self.final_act_doa = nn.Tanh()

    def init_weight(self):
        pass


    def forward(self, x):

        # list of layers in shared space
        shared_feature = [[0]*7 for _ in range(4)]
        #shared_feature_block_2 = [[0]  for _ in range(4)]
        shared = x

        for j, share in enumerate(shared_feature): # iterate over the shared blocks
            for l,layer in enumerate(share): # iterate over the layers of the shared blocks
                shared_feature[j][l] = self.shared_network[j][l](shared)
                shared = shared_feature[j][l] # update the shared

        # define attention list for tasks
        atten_encoder, atten_decoder = ([0] * 3 for _ in range(2))
        for i in range(3):
            atten_encoder[i], atten_decoder[i] = ([0] * 5 for _ in range(2))
        for i in range(3):
            for j in range(5):
                atten_encoder[i][j], atten_decoder[i][j] = ([0] * 3 for _ in range(2))

        for i in range(2): # iterate over the tasks
            for j in range(4): # iterate over the shared feature space
                if j == 0:
                    # a
                    atten_encoder[i][j][0] = self.encoder_att[i][j](shared_feature[j][0])
                    # a_hat
                    atten_encoder[i][j][1] = (atten_encoder[i][j][0]) * shared_feature[j][3]
                    # f function
                    atten_encoder[i][j][2] = self.encoder_block_att[j](atten_encoder[i][j][1])
                    # maxpooling
                    atten_encoder[i][j][2] = F.avg_pool2d(atten_encoder[i][j][2], kernel_size=(2, 2), stride=2)
                else:
                    if (j == 3):
                        atten_encoder[i][j][0] = self.encoder_att[i][j](
                            torch.cat((shared_feature[j][0], atten_encoder[i][j - 1][2]), dim=1))
                        atten_encoder[i][j][1] = (atten_encoder[i][j][0]) * shared_feature[j][3]
                        atten_encoder[i][j][2] = self.encoder_block_att[j](atten_encoder[i][j][1])
                    elif (j == 2):
                        atten_encoder[i][j][0] = self.encoder_att[i][j](
                            torch.cat((shared_feature[j][0], atten_encoder[i][j - 1][2]), dim=1))
                        atten_encoder[i][j][1] = (atten_encoder[i][j][0]) * shared_feature[j][3]
                        atten_encoder[i][j][2] = self.encoder_block_att[j](atten_encoder[i][j][1])
                        atten_encoder[i][j][2] = F.avg_pool2d(atten_encoder[i][j][2], kernel_size=(1, 2))
                    else:
                        atten_encoder[i][j][0] = self.encoder_att[i][j](
                            torch.cat((shared_feature[j][0], atten_encoder[i][j - 1][2]), dim=1))
                        atten_encoder[i][j][1] = (atten_encoder[i][j][0]) * shared_feature[j][3]
                        atten_encoder[i][j][2] = self.encoder_block_att[j](atten_encoder[i][j][1])
                        atten_encoder[i][j][2] = F.avg_pool2d(atten_encoder[i][j][2], kernel_size=(1,2), stride=2)

        x_sed = atten_encoder[0][-2][-1].mean(dim=3)  # (N, C, T)
        x_doa = atten_encoder[1][-2][-1].mean(dim=3)  # (N, C, T)

        # transformer
        if self.pe_enable:
            x_sed = self.pe(x_sed)
        if self.pe_enable:
            x_doa = self.pe(x_sed)
        x_sed = x_sed.permute(2, 0, 1)  # (T, N, C)
        x_doa = x_doa.permute(2, 0, 1)  # (T, N, C)

        x_sed_1 = self.sed_trans_track1(x_sed).transpose(0, 1)  # (N, T, C)
        x_sed_2 = self.sed_trans_track2(x_sed).transpose(0, 1)  # (N, T, C)
        x_doa_1 = self.doa_trans_track1(x_doa).transpose(0, 1)  # (N, T, C)
        x_doa_2 = self.doa_trans_track2(x_doa).transpose(0, 1)  # (N, T, C)

        # fc
        x_sed_1 = self.final_act_sed(self.fc_sed_track1(x_sed_1))
        x_sed_2 = self.final_act_sed(self.fc_sed_track2(x_sed_2))
        x_sed = torch.stack((x_sed_1, x_sed_2), 2)
        x_doa_1 = self.final_act_doa(self.fc_doa_track1(x_doa_1))
        x_doa_2 = self.final_act_doa(self.fc_doa_track2(x_doa_2))
        x_doa = torch.stack((x_doa_1, x_doa_2), 2)

        output = {
            'sed': x_sed,
            'doa': x_doa,
        }
        return output

    def att_layer(self, channel):
        '''
        g and h functions in the paper

        Args:
            channel: list of filters
            Returns: att_block
        '''
        att_block = nn.Sequential(
            nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=1, padding=0),
            nn.BatchNorm2d(channel[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channel[1], out_channels=channel[2], kernel_size=1, padding=0),
            nn.BatchNorm2d(channel[2]),
            nn.Sigmoid(),
        )
        return att_block

    def conv_layer(self, channel, pred=False):
        '''
        f function in the paper
        Args:
            channel:
            pred:

        Returns:

        '''
        if not pred:
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=channel[1]),
                nn.ReLU(inplace=True),
            )
        else:
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels=channel[0], out_channels=channel[0], kernel_size=3, padding=1),
                nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=1, padding=0),
            )
        return conv_block