import librosa
from utils.common import int16_samples_to_float32


def apply_pitch_shift (x, sample_rate):

    x = int16_samples_to_float32(x)
    x[0, :]  = librosa.effects.pitch_shift(x[0,:],sample_rate,  n_steps=4)
    x[1, :] = librosa.effects.pitch_shift(x[1, :], sample_rate, n_steps=4)
    x[2, :] = librosa.effects.pitch_shift(x[2, :], sample_rate, n_steps=4)
    x[3, :] = librosa.effects.pitch_shift(x[3, :], sample_rate, n_steps=4)

    return x
