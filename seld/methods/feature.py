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
import matplotlib.cm as cm
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import librosa

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

    def define_transformation(self,waveform):
        sample_rate = 24000
        n_fft = 1024
        win_length = None
        hop_length = 600
        n_mels = 256

        mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            power=2.0,
            n_mels=n_mels,
        )

        melspec = mel_spectrogram(waveform.cpu())

        return melspec


    def forward(self, x):
        """
        input:
            (batch_size, channels=4, data_length)
        output:
            (batch_size, channels, time_steps, freq_bins)
        """
        # for infrerence

        if type(x)!= tuple :
            if x.ndim == 3:
                x = self.stft_extractor(x)
                logmel = self.logmel_extractor(self.spectrogram_extractor(x))
                intensity_vector = self.intensityVector_extractor(x, self.logmel_extractor.melW)
                out = torch.cat((logmel, intensity_vector), dim=1)
                return out
            else:
                raise ValueError("x shape must be (batch_size, num_channels, data_length)\n \
                                                    Now it is {}".format(x.shape))

        input, target, ind, data_type = x
        if input.ndim != 3:
            raise ValueError("x shape must be (batch_size, num_channels, data_length)\n \
                            Now it is {}".format(input.shape))
        #self.plot_waveform(input[0])
        #melspec = self.define_transformation(input[0])
        #self.plot_spectrogram(melspec)
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

    def plot_spectrogram(self, spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
        fig, axs = plt.subplots(1, 1)
        axs.set_title(title or 'Spectrogram (db)')
        axs.set_ylabel(ylabel)
        axs.set_xlabel('frame')
        im = axs.imshow(librosa.power_to_db(spec[0]), origin='lower', aspect=aspect)
        if xmax:
            axs.set_xlim((0, xmax))
        fig.colorbar(im, ax=axs)
        plt.show(block=False)
        plt.savefig('Spectrogram.png', format='png')
        plt.close(fig)

    def plot_waveform(self,waveform, title="Waveform", xlim=None, ylim=None):
        num_channels, num_frames = waveform.shape
        time_axis = torch.arange(0, num_frames)
        # // sample_rate
        figure, axes = plt.subplots(num_channels, 1)
        if num_channels == 1:
            axes = [axes]
        for c in range(num_channels):
            axes[c].plot(time_axis, waveform[c].cpu(), linewidth=1)
            axes[c].grid(True)
            if num_channels > 1:
                axes[c].set_ylabel(f'Channel {c + 1}')
            if xlim:
                axes[c].set_xlim(xlim)
            if ylim:
                axes[c].set_ylim(ylim)
        figure.suptitle(title)
        plt.show(block=False)
        plt.savefig('waveform.png', format='png')
        plt.close(figure)

    '''
    # For spectrogram visualization
    def plot_specgram(self,waveform, sample_rate, title="Spectrogram", xlim=None):
        #waveform = waveform[0].numpy()
        waveform = waveform[0].cpu().numpy()

        num_channels, num_frames = waveform.shape
        time_axis = torch.arange(0, num_frames) // sample_rate

        figure, axes = plt.subplots(num_channels, 1)
        if num_channels == 1:
            axes = [axes]
        for c in range(num_channels):
            axes[c].specgram(waveform[c], Fs=sample_rate)
            if num_channels > 1:
                axes[c].set_ylabel(f'Channel {c + 1}')
            if xlim:
                axes[c].set_xlim(xlim)
        figure.suptitle(title)
        plt.savefig('Spec')
        plt.show(block=False)
    
    '''