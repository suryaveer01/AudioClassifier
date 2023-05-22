import json
import torch
from torchvision import models, transforms
from PIL import Image as PilImage

import matplotlib.pyplot as plt

import numpy as np

import torch.nn as nn

from torch.utils.data import DataLoader
from urban_dataset import UrbanSoundDataset


import torchaudio

from omnixai.data.image import Image
from omnixai.explainers.vision.specific.gradcam.pytorch.gradcam import GradCAM


ANNOTATIONS_FILE = ".data/UrbanSound8K/metadata/UrbanSound8K.csv"
AUDIO_DIR = ".data/UrbanSound8K/audio"
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050
device = 'cpu'


# Load the test image
img = Image(PilImage.open('output.png'))
# img = Image(PilImage.open('cat.jpg').convert('RGB'))

# Load the class names
# with open('../data/images/imagenet_class_index.json', 'r') as read_file:
#     class_idx = json.load(read_file)
#     idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]

idx2label = ['cat','dog','frog','horse','ship','truck']

def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader
# instantiating our dataset object and create data loader
mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=1024,
    hop_length=512,
    n_mels=64
)
mfcc_transform = torchaudio.transforms.MFCC(sample_rate=SAMPLE_RATE,n_mfcc=64)

spectogram_transform = torchaudio.transforms.Spectrogram(
    n_fft=1024,
    hop_length=512)

usd = UrbanSoundDataset(ANNOTATIONS_FILE,
                        AUDIO_DIR,
                        mfcc_transform,
                        SAMPLE_RATE,
                        NUM_SAMPLES,
                        device)

print((usd[0]))
# img = Image(usd[0])
# A ResNet Model
model = models.resnet34()

# Load the model weights
model.fc = nn.Linear(512,10)
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

model.load_state_dict(torch.load('models/resnet34_50_MFCC.pth'))
# The preprocessing model
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[ 0.406], std=[ 0.225])
])
# preprocess = lambda ims: torch.stack([transform(im.to_pil()) for im in ims])

preprocess = lambda ims: torch.stack([transform(im.to_pil()) for im in ims])

explainer = GradCAM(
    model=model,
    target_layer=model.layer4[-1],
    preprocess_function=preprocess
)
# Explain the top label
explanations = explainer.explain(img)
explanations.ipython_plot(index=0, class_names=idx2label)