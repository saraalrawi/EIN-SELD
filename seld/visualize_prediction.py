########################################################################
# This script provided by the DCASE orginizers, we are just adapting it according to our needs.
########################################################################
import os
import sys
import argparse
import numpy as np
import librosa.display
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plot
import scipy.io.wavfile as wav
from pathlib import Path



parser = argparse.ArgumentParser()

parser.add_argument('--dataset_dir',help='dataset_root directory', default= '/home/alrawis/EIN-SELD/_dataset/dataset_root/')
parser.add_argument('--pred_file',help='Prediction file to visualize', default= '/home/alrawis/EIN-SELD/submissions/mix001.csv')
parser.add_argument('--plot_loc',help='Location to save the plot', default= '/home/alrawis/EIN-SELD/submissions/')

args = parser.parse_args()
print('The script is running with following args:\n')
print(args)



plot.switch_backend('agg')
plot.rcParams.update({'font.size': 16})



#classes in the FOA dataset
classes = {
    'alarm': 0,
    'baby': 1,
    'crash': 2,
    'dog': 3,
    'engine': 4,
    'female_scream': 5,
    'female_speech': 6,
    'fire': 7,
    'footsteps': 8,
    'knock': 9,
    'male_scream': 10,
    'male_speech': 11,
    'phone': 12,
    'piano': 13
}
# the number of audio channels
nb_channels = 4
# sampling rate
fs=24000
#  audio max length is 60 seconds
max_audio_len_s = 60 * fs
_eps = 1e-8
n_fft = 64
hop_length = 600
label_resol = 0.1
# number of classes in the dataset
nb_classes = len(classes)
# change to your directory
dataset_dir = args.dataset_dir
# prediction file to visualize /submissions is the default file for saving the predictions, predictions are saved per file.
pred = Path(args.pred_file)
# I am saving the plot in the same location with as jpg
save_loc = Path(args.plot_loc)

#########################################################################################################
def load_output_format_file(_output_format_file):
    """
    Loads DCASE output format csv file and returns it in dictionary format
    :param _output_format_file: DCASE output format CSV
    :return: _output_dict: dictionary
    """
    _output_dict = {}
    _fid = open(_output_format_file, 'r')
    # next(_fid)
    for _line in _fid:
        _words = _line.strip().split(',')
        _frame_ind = int(_words[0])
        if _frame_ind not in _output_dict:
            _output_dict[_frame_ind] = []

        if len(_words) == 4:  # read polar coordinates format, we ignore the track count, for our prediction
            _output_dict[_frame_ind].append([int(_words[1]), float(_words[2]), float(_words[3])])
        if len(_words) == 5:  # read polar coordinates format, we ignore the track count, for the groundtruth
            _output_dict[_frame_ind].append([int(_words[1]), float(_words[3]), float(_words[4])])
        elif len(_words) == 6:  # read Cartesian coordinates format, we ignore the track count
            _output_dict[_frame_ind].append([int(_words[1]), float(_words[3]), float(_words[4]), float(_words[5])])
    _fid.close()
    return _output_dict

def collect_classwise_data(_in_dict):
    _out_dict = {}
    for _key in _in_dict.keys():
        for _seld in _in_dict[_key]:
            if _seld[0] not in _out_dict:
                _out_dict[_seld[0]] = []
            _out_dict[_seld[0]].append([_key, _seld[0], _seld[1], _seld[2]])
    return _out_dict

def _load_audio(audio_path):
    fs, audio = wav.read(audio_path)
    audio = audio[:, :nb_channels] / 32768.0 + _eps
    if audio.shape[0] <  max_audio_len_s:
        zero_pad = np.random.rand(max_audio_len_s - audio.shape[0], audio.shape[1])*_eps
        audio = np.vstack((audio, zero_pad))
    elif audio.shape[0] > max_audio_len_s:
        audio = audio[: max_audio_len_s, :]
    return audio, fs

def _next_greater_power_of_2(x):
    return 2 ** (x - 1).bit_length()
_win_len = 2 * hop_length
_nfft = _next_greater_power_of_2(_win_len)
_max_feat_frames = int(np.ceil(max_audio_len_s / hop_length))

def _spectrogram(audio_input):
    _nb_ch = audio_input.shape[1]
    nb_bins = _nfft // 2
    #nb_bins =  256
    spectra = np.zeros((_max_feat_frames, nb_bins + 1, _nb_ch), dtype=complex)
    for ch_cnt in range(_nb_ch):
        stft_ch = librosa.core.stft(np.asfortranarray(audio_input[:, ch_cnt]), n_fft=_nfft, hop_length=hop_length,
                                        win_length=_win_len, window='hann')
        spectra[:, :, ch_cnt] = stft_ch[:, :_max_feat_frames].T
    return spectra


def plot_func(plot_data, hop_len_s, ind, plot_x_ax=False, plot_y_ax=False):
    cmap = ['b', 'r', 'g', 'y', 'k', 'c', 'm', 'b', 'r', 'g', 'y', 'k', 'c', 'm']
    for class_ind in plot_data.keys():
        time_ax = np.array(plot_data[class_ind])[:, 0] * hop_len_s
        y_ax = np.array(plot_data[class_ind])[:, ind]
        plot.plot(time_ax, y_ax, marker='.', color=cmap[class_ind], linestyle='None', markersize=4)
    plot.grid()
    plot.xlim([0, 60])
    if not plot_x_ax:
        plot.gca().axes.set_xticklabels([])

    if not plot_y_ax:
        plot.gca().axes.set_yticklabels([])
##########################################################################################################

# path of the audio directory for visualizing the spectrogram and description directory for
# visualizing the reference
# Note: The code finds out the audio filename from the predicted filename automatically
ref_dir = os.path.join(dataset_dir, 'metadata_eval')
aud_dir = os.path.join(dataset_dir, 'foa_eval')

# convert csv file to a dict
pred_dict = load_output_format_file(pred)

# get the ground truth filename
ref_filename = os.path.basename(pred)
# convert csv to dict
ref_dict_polar = load_output_format_file(os.path.join(ref_dir, ref_filename))

# the dcase output format is in polar, our pred is already in polar
pred_data = collect_classwise_data(pred_dict)
ref_data = collect_classwise_data(ref_dict_polar)

# we need the wave form to generate the spectrogram
ref_filename = os.path.basename(pred).replace('.csv', '.wav')
audio, fs = _load_audio(os.path.join(aud_dir, ref_filename))
stft = np.abs(np.squeeze(_spectrogram(audio[:, :1])))
stft = librosa.amplitude_to_db(stft, ref=np.max)

############################################################################################################################

plot.figure(figsize=(20, 10))
gs = gridspec.GridSpec(4, 4)
ax0 = plot.subplot(gs[0, 1:3]), librosa.display.specshow(stft.T, sr=fs , hop_length=600, x_axis='s', y_axis='linear'), plot.xlim([0, 60]), plot.xticks([]), plot.xlabel(''), plot.title('Spectrogram')
ax1 = plot.subplot(gs[1, :2]), plot_func(ref_data, label_resol, ind=1, plot_y_ax=True), plot.ylim([-1, nb_classes + 1]), plot.title('SED Groundtruth')
ax2 = plot.subplot(gs[1, 2:]), plot_func(pred_data, label_resol, ind=1), plot.ylim([-1, nb_classes + 1]), plot.title('SED Predicted')
ax3 = plot.subplot(gs[2, :2]), plot_func(ref_data, label_resol, ind=2, plot_y_ax=True), plot.ylim([-180, 180]), plot.title('Azimuth SED Groundtruth')
ax4 = plot.subplot(gs[2, 2:]), plot_func(pred_data, label_resol, ind=2), plot.ylim([-180, 180]), plot.title('Azimuth Predicted')
ax5 = plot.subplot(gs[3, :2]), plot_func(ref_data, label_resol, ind=3, plot_y_ax=True), plot.ylim([-90, 90]), plot.title('Elevation SED Groundtruth')
ax6 = plot.subplot(gs[3, 2:]), plot_func(pred_data, label_resol, ind=3), plot.ylim([-90, 90]), plot.title('Elevation Predicted')
ax_lst = [ax0, ax1, ax2, ax3, ax4, ax5, ax6]
plot.savefig(os.path.join(save_loc , ref_filename.replace('.wav', '.jpg')), dpi=300, bbox_inches = "tight")