import os
import random
import numpy as np

import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import matplotlib.pyplot as plt

import skimage.io


class UrbanSoundDataset(Dataset):

    def __init__(self,
                 annotations_file,
                 audio_dir,
                 transformation,
                 mfcc_transformation,
                 target_sample_rate,
                 num_samples,
                 device):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation.to(self.device)
        self.mfcc_transformation = mfcc_transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        wave_form = signal
        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal_mel = self.transformation(signal)
        signal_mfcc = self.mfcc_transformation(signal)
        return signal_mel, label ,wave_form , signal_mfcc

    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _get_audio_sample_path(self, index):
        fold = f"fold{self.annotations.iloc[index, 5]}"
        path = os.path.join(self.audio_dir, fold, self.annotations.iloc[
            index, 0])
        return path

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 6]
    
def plot_spectrogram(specgram, title=None, ylabel="freq_bin"):
    plt.close()
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(torch.log10(specgram) * 10, origin="lower", aspect="auto")
    fig.colorbar(im, ax=axs)
    plt.savefig("spectograms/"f"{title}.png")
    # plt.show(block=True)

def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled

def plot_mfcc_spectrogram(specgram, title=None, ylabel="freq_bin"):
    
    # assuming specgram is a tensor object
    specgram_np = specgram.numpy()

    # scale the values to [0, 255] range and convert to uint8 datatype
    img = np.uint8(255 * (specgram_np - np.min(specgram_np)) / (np.max(specgram_np) - np.min(specgram_np)))

    # flip the image along the y-axis
    img = np.flip(img, axis=0)

    # invert the image
    # img = 255 - img

    # save as PNG
    skimage.io.imsave('output.png', img)
    
    plt.close()
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(specgram, origin="lower", aspect="auto")
    fig.colorbar(im, ax=axs)
    plt.savefig("spectograms/"f"{title}.png")
    # plt.show(block=True)
    # plt.savefig(f"spectograms/{title}.png")

def plot_wavefrom(waveform, title=None, ylabel="Amplitude"):
    print(waveform.shape)
    plt.close()
    plt.plot(waveform.t().numpy())
    plt.title(title or "Waveform")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.savefig("spectograms/"f"{title}.png")
    # plt.show(block=True)


if __name__ == "__main__":
    ANNOTATIONS_FILE = ".data/UrbanSound8K/metadata/UrbanSound8K.csv"
    AUDIO_DIR = ".data/UrbanSound8K/audio"
    SAMPLE_RATE = 22050
    NUM_SAMPLES = 22050

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device {device}")

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    mfcc_transform = torchaudio.transforms.MFCC(sample_rate=SAMPLE_RATE,n_mfcc=20,melkwargs=dict(n_fft=200, n_mels=64))

    spectogram_transform = torchaudio.transforms.Spectrogram(
        n_fft=1024,
        hop_length=512)

    # usd = UrbanSoundDataset(ANNOTATIONS_FILE,
    #                         AUDIO_DIR,
    #                         mel_spectrogram,
    #                         SAMPLE_RATE,
    #                         NUM_SAMPLES,
    #                         device)
    
    # usd = UrbanSoundDataset(ANNOTATIONS_FILE,
    #                         AUDIO_DIR,
    #                         mfcc_transform,
    #                         SAMPLE_RATE,
    #                         NUM_SAMPLES,
    #                         device)

    usd = UrbanSoundDataset(ANNOTATIONS_FILE,
                            AUDIO_DIR,
                            mel_spectrogram,
                            mfcc_transform,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            device)
    
    print(f"There are {len(usd)} samples in the dataset.")
    print(str(torchaudio.get_audio_backend()))
    class_names = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music"] 
    random_indexes = random.sample(range(1, 8733), 20)
    for e in random_indexes:
        signal, label ,wave_form, mfcc_signal = usd[e]
        # plt.imshow(signal[0], interpolation="nearest", origin="lower", aspect="auto")
        # # plt.colorbar()
        # plt.show()
        plot_spectrogram(signal[0], title=f"Mel_Spectrogram_torchaudio_{class_names[label]}", ylabel="mel freq")
        plot_mfcc_spectrogram(mfcc_signal[0], title=f"MFCC_torchaudio_{class_names[label]}", ylabel="MFCC")
        plot_wavefrom(wave_form, title=f"WaveFrom_torchaudio_{class_names[label]}", ylabel="mel freq")
