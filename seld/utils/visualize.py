import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import librosa
from scipy.io.arff.tests.test_arffread import data_path


def plot_spec(data: np.array, sr: int, title: str, fpath: str) -> None:
    '''
    Function for plotting spectrogram along with amplitude wave graph
    '''
    label = str(fpath).split('/')[-1].split('_')[0]
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].title.set_text(f'{title} / Label: {label}')
    ax[0].specgram(data, Fs=2)
    ax[1].set_ylabel('Amplitude')
    ax[1].plot(np.linspace(0, 1, len(data)), data)


# Reading the wav file:
file_path = data_path.ls()[3]
wav, sr = librosa.load(file_path, sr=None)
plt.savefig()

# Plotting the spectrogram and wave graph, calling the function
#plot_spec(wav, sr, 'Original wave file', file_path)