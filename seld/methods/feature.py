import torch
import torch.nn as nn
from methods.ein_seld.data_augmentation import spec_augment_
from methods.ein_seld.data_augmentation import spec_augment, channel_rotation
from methods.utils.stft import (STFT, LogmelFilterBank, intensityvector,
                                spectrogram_STFTInput)
import numpy as np

class LogmelIntensity_Extractor(nn.Module):
    def __init__(self, cfg , data_type):
        super().__init__()

        data = cfg['data']
        sample_rate, n_fft, hop_length, window, n_mels, fmin, fmax = \
            data['sample_rate'], data['n_fft'], data['hop_length'], data['window'], data['n_mels'], \
                data['fmin'], data['fmax']
        

        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # STFT extractor
        self.stft_extractor = STFT(n_fft=n_fft, hop_length=hop_length, win_length=n_fft, 
            window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=data['feature_freeze'])
        
        # Spectrogram extractor
        self.spectrogram_extractor = spectrogram_STFTInput
        
        # Logmel extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=n_fft, 
            n_mels=n_mels, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=data['feature_freeze'])

        # Intensity vector extractor
        self.intensityVector_extractor = intensityvector

        self.data_type = data_type


    def forward(self, x):
        """
        input:
            (batch_size, channels=4, data_length)
        output:
            (batch_size, channels, time_steps, freq_bins)
        """
        input, target, ind, data_type = x
        if input.ndim != 3:
            raise ValueError("x shape must be (batch_size, num_channels, data_length)\n \
                            Now it is {}".format(input.shape))
        # get the indices of augmented data
        aug_idx_spc = [i for i, x in enumerate(data_type) if x == "train_spec_aug"]

        '''
        if ind == 'train':
            if np.random.random() > 0.5:
                input[:, :], pattern = channel_rotation.apply_data_channel_rotation('foa', input[:, :])
                target['doa'] = channel_rotation.apply_label_channel_rotation('foa', target['doa'], pattern)
        '''
        aug_idx_rotate = [i for i, x in enumerate(data_type) if x == "train_rotate_channel"]
        if ind == 'train'  and len(aug_idx_rotate) != 0:
            for i , dt in enumerate(aug_idx_rotate):
                input[i, :, :], pattern = channel_rotation.apply_data_channel_rotation('foa', input[i, :, :])
                aug_rotate = channel_rotation.apply_label_channel_rotation('foa', target['doa'][i], pattern)
                # update the target
                target['doa'][i] = aug_rotate


        input = self.stft_extractor(input)
        logmel = self.logmel_extractor(self.spectrogram_extractor(input))


        if ind == 'train' and len(aug_idx_spc) != 0:
            for i , dt in enumerate(aug_idx_spc):
                logmel[dt, :, :, :] = spec_augment.specaug(torch.squeeze(logmel[dt,:,:,:]).permute(0, 2, 1))

        intensity_vector = self.intensityVector_extractor(input, self.logmel_extractor.melW)
        out = torch.cat((logmel, intensity_vector), dim=1)
        return out, target

    '''
    def forward(self, x):
        """
        input: 
            ((batch_size, channels=4, data_length), target ,data_type)
        output: 
            (batch_size, channels, time_steps, freq_bins)
        """
        input, target, ind, data_type = x
        if input.ndim != 3:
            raise ValueError("x shape must be (batch_size, num_channels, data_length)\n \
                            Now it is {}".format(input.shape))

        if ind == 'train':
            for i , dt in enumerate(data_type):
                if dt == 'train_rotate_channel':
                    input[i, :, :], pattern = channel_rotation.apply_data_channel_rotation('foa',input[i, :, :])
                    target['doa'][i] = channel_rotation.apply_label_channel_rotation('foa', target['doa'][i], pattern)

        input = self.stft_extractor(input)
        logmel = self.logmel_extractor(self.spectrogram_extractor(input))

        if ind == 'train':
            for i , dt in enumerate(data_type):
                if dt == 'train_spec_aug':
                    logmel_i = spec_augment.specaug(torch.squeeze(logmel[i,:,:,:]).permute(0, 2, 1))
                    logmel[i,:,:,:] = logmel_i

        intensity_vector = self.intensityVector_extractor(input, self.logmel_extractor.melW)
        out = torch.cat((logmel, intensity_vector), dim=1)
        return (out, target)
    '''