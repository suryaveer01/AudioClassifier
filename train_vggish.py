import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader

from dataset import UrbanSoundDataset
from audioclassifiernetwork import CNNNetwork

import torchaudio.models as models


BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001

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


def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader


def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
    for input, target in data_loader:
        input, target = input.to(device), target.to(device)
        
        total_loss = 0
        correct = 0
        total = 0
        # calculate loss
        prediction = model(input)
        loss = loss_fn(prediction, target)
        total_loss += loss.item()

        # backpropagate error and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        _, predicted = prediction.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    accuracy = correct / total

    print(f"loss: {loss.item()} Accuracy: {accuracy}")


def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_single_epoch(model, data_loader, loss_fn, optimiser, device)
        print("---------------------------")
    print("Finished training")


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")

    # instantiating our dataset object and create data loader
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    usd = UrbanSoundDataset(ANNOTATIONS_FILE,
                            AUDIO_DIR,
                            mel_spectrogram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            device)
    
    train_dataloader = create_data_loader(usd, BATCH_SIZE)

    # construct model and assign it to device
    # cnn = CNNNetwork().to(device)
    # # print(cnn)

    # Define the PyTorch model
    vggish_model = models.vggish(pretrained=False)

    # Load the weights from the TensorFlow checkpoint file
    state_dict = torch.load('vggish_model.ckpt', map_location=torch.device('cpu'))
    vggish_model.load_state_dict(state_dict)
    vggish_model = vggish_model.to(device)
    # initialise loss funtion + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(vggish_model.parameters(),
                                 lr=LEARNING_RATE)

    # train model
    train(vggish_model, train_dataloader, loss_fn, optimiser, device, EPOCHS)

    # save model
    torch.save(vggish_model.state_dict(), "vggish_net.pth")
    print("Trained feed forward net saved at feedforwardnet.pth")