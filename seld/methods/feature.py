import torch
import torch.nn as nn
from methods.ein_seld.data_augmentation import spec_augment_
from methods.ein_seld.data_augmentation import spec_augment, channel_rotation
from methods.utils.stft import (STFT, LogmelFilterBank, intensityvector,
                                spectrogram_STFTInput)
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

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
        self.cfg = cfg


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
        aug_idx_inverse = [i for i, x in enumerate(data_type) if x == "train_invert_position_aug"]
        if ind == 'train' and len(aug_idx_inverse) != 0:
            for i, dt in enumerate(aug_idx_inverse):
                input[i, :, :] = torch.flip(input[i, :, :], dims=[1])  # invert waveform time axis
                sed_label = torch.flip(target['sed'][i], dims=[0])  # invert sed label time axis
                doa_label = torch.flip(target['doa'][i], dims=[0])  # invert doa label time axis
                doa_label = 0.0 - doa_label  # also invert sound source position
                target['sed'][i] = sed_label
                target['doa'][i] = doa_label

        aug_idx_rotate = [i for i, x in enumerate(data_type) if x == "train_rotate_channel"]
        if ind == 'train'  and len(aug_idx_rotate) != 0:
            for i , dt in enumerate(aug_idx_rotate):
                input[i, :, :], pattern = channel_rotation.apply_data_channel_rotation('foa', input[i, :, :])
                aug_rotate = channel_rotation.apply_label_channel_rotation('foa', target['doa'][i], pattern)
                # update the target
                target['doa'][i] = aug_rotate


        input = self.stft_extractor(input)
        logmel = self.logmel_extractor(self.spectrogram_extractor(input))

        aug_idx_spc = [i for i, x in enumerate(data_type) if x == "train_spec_aug"]

        if ind == 'train' and len(aug_idx_spc) != 0:
            # get specAugment Parameters
            F = self.cfg['data_augmentation']['F']
            T = self.cfg['data_augmentation']['T']
            num_freq_masks = self.cfg['data_augmentation']['num_freq_masks']
            num_time_masks = self.cfg['data_augmentation']['num_time_masks']
            replace_with_zero = self.cfg['data_augmentation']['replace_with_zero']

            for i , dt in enumerate(aug_idx_spc):

                logmel_aug = spec_augment.specaug(torch.squeeze(logmel[dt,:,:,:]).permute(0, 2, 1),
                                                  W=2, F=F, T=T,
                                                  num_freq_masks=num_freq_masks,
                                                  num_time_masks=num_time_masks,
                                                  replace_with_zero=replace_with_zero)
                logmel[dt, :, :, :] = logmel_aug
        intensity_vector = self.intensityVector_extractor(input, self.logmel_extractor.melW)
        out = torch.cat((logmel, intensity_vector), dim=1)
        return out, target

    # For spectrogram visualization
    def plot_spectrogram(self, spect):
        spect_cpu = spect.cpu()
        librosa.display.specshow(librosa.power_to_db(spect_cpu[2,:,:]**2, ref=np.max),sr = 24000, y_axis = 'log', x_axis = 'time')

        plt.title('Mel Spectrogram')
        plt.colorbar(format='%+2.0f dB')
        plt.savefig('spec_channel_1.png')