import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader

from urban_dataset import UrbanSoundDataset

from torchvision.models import resnet34

from torch.utils.tensorboard import SummaryWriter


writer = SummaryWriter()


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
    sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=64
)


def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader


def train_single_epoch(model, data_loader, loss_fn, optimiser, device, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for input, target in data_loader:
        input, target = input.to(device), target.to(device)

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
    average_loss = total_loss / len(data_loader)

    writer.add_scalar("Train/Loss", average_loss, epoch)
    writer.add_scalar("Train/Accuracy", accuracy, epoch)

    print(f"Trainig loss: {average_loss:.3f} Accuracy: {accuracy:.3f}")


def validate_single_epoch(model, data_loader, loss_fn, device, epoch):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for input, target in data_loader:
            input, target = input.to(device), target.to(device)
            # calculate loss
            prediction = model(input)
            loss = loss_fn(prediction, target)
            total_loss += loss.item()

            _, predicted = prediction.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    accuracy = correct / total
    average_loss = total_loss / len(data_loader)

    writer.add_scalar("Validation/Loss", average_loss, epoch)
    writer.add_scalar("Validation/Accuracy", accuracy, epoch)

    print(f"Validation loss: {average_loss:.3f} Accuracy: {accuracy:.3f}")
    return accuracy, average_loss


def train(model, train_dataloader, test_dataloader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_single_epoch(
            model, train_dataloader, loss_fn, optimiser, device, epoch=i + 1
        )
        validate_single_epoch(model, test_dataloader, loss_fn, device, epoch=i + 1)
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
        sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=64
    )
    mfcc_transform = torchaudio.transforms.MFCC(sample_rate=SAMPLE_RATE, n_mfcc=64)

    spectogram_transform = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=512)

    usd = UrbanSoundDataset(
        ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES, device
    )

    train_test_split = 0.2
    train_size = int(len(usd) * (1 - train_test_split))
    test_size = len(usd) - train_size
    train_usd, test_usd = torch.utils.data.random_split(usd, [train_size, test_size])

    train_dataloader = create_data_loader(train_usd, BATCH_SIZE)
    test_dataloader = create_data_loader(test_usd, BATCH_SIZE)

    # # construct model and assign it to device
    # cnn = CNNNetwork().to(device)
    # print(cnn)

    resnet_model = resnet34()
    resnet_model.fc = nn.Linear(512, 10)
    resnet_model.conv1 = nn.Conv2d(
        1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )
    resnet_model = resnet_model.to(device)

    # initialise loss funtion + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(
        resnet_model.parameters(), lr=LEARNING_RATE, weight_decay=0.0001
    )

    # train model
    train(
        resnet_model,
        train_dataloader,
        test_dataloader,
        loss_fn,
        optimiser,
        device,
        EPOCHS,
    )

    # save model
    torch.save(resnet_model.state_dict(), "resnet34_50_MEL_SPECTROGRAM.pth")
    print("Trained feed forward net saved at feedforwardnet.pth")
    writer.close()
