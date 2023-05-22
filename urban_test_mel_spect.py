import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader

from urban_dataset import UrbanSoundDataset

from torchvision.models import resnet34

from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns


writer = SummaryWriter()


BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001

ANNOTATIONS_FILE = ".data/UrbanSound8K/metadata/UrbanSound8K.csv"
AUDIO_DIR = ".data/UrbanSound8K/audio"
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050

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
    "street_music",
]

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
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    for input, target in data_loader:
        y_true.append(target)

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
        y_pred.extend(predicted.numpy())

    accuracy = correct / total
    average_loss = total_loss / len(data_loader)

    writer.add_scalar("Train/Loss", average_loss, epoch)
    writer.add_scalar("Train/Accuracy", accuracy, epoch)
    cm = confusion_matrix(y_true, y_pred)
    print("cm", cm)

    print(f"Trainig loss: {average_loss:.3f} Accuracy: {accuracy:.3f}")


def test_single_epoch(model, data_loader, loss_fn, device, epoch):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for input, target in data_loader:
            y_true.extend(target)
            input, target = input.to(device), target.to(device)
            # calculate loss
            prediction = model(input)
            loss = loss_fn(prediction, target)
            total_loss += loss.item()

            _, predicted = prediction.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            y_pred.extend(predicted.numpy())

    accuracy = correct / total
    average_loss = total_loss / len(data_loader)
    cm = confusion_matrix(y_true, y_pred)
    print(cm, "cm")

    sns.set()
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        cmap="Blues",
        fmt="g",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    # plt.show()

    writer.add_scalar("Test/Loss", average_loss, epoch)
    writer.add_scalar("Test/Accuracy", accuracy, epoch)

    print(f"Test loss: {average_loss:.3f} Test: {accuracy:.3f}")
    return accuracy, average_loss


# def train(model, train_dataloader, test_dataloader, loss_fn, optimiser, device, epochs):
#     for i in range(epochs):
#         print(f"Epoch {i+1}")
#         train_single_epoch(model, train_dataloader, loss_fn, optimiser, device,epoch=i+1)
#         test_single_epoch(model, test_dataloader, loss_fn, device,epoch=i+1)
#         print("---------------------------")
#     print("Finished training")


def test(model, train_dataloader, test_dataloader, loss_fn, device, epochs):
    final_accuracy = 0
    final_loss = 0
    for i in range(epochs):
        print(f"Epoch {i+1}")
        accuracy, average_loss = test_single_epoch(
            model, test_dataloader, loss_fn, device, epoch=i + 1
        )
        final_accuracy += accuracy
        final_loss += average_loss
        print("---------------------------")
    print(f"Test loss: {final_loss/epochs:.3f} Test: {final_accuracy/epochs:.3f}")
    print("Finished Testing")


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

    resnet_model.load_state_dict(
        torch.load(
            "models/resnet34_50_MEL_SPECTROGRAM.pth", map_location=torch.device(device)
        )
    )
    resnet_model.eval()

    # initialise loss funtion + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(resnet_model.parameters(), lr=LEARNING_RATE)

    # train model
    # train(resnet_model, train_dataloader,test_dataloader, loss_fn, optimiser, device, EPOCHS)

    # test model
    test(resnet_model, train_dataloader, test_dataloader, loss_fn, device, EPOCHS)
    # test_accuracy, test_loss = validate_single_epoch(resnet_model, test_dataloader, loss_fn, device,epoch=1)

    # save model
    # torch.save(resnet_model.state_dict(), "feedforwardnet.pth")
    # print("Trained feed forward net saved at feedforwardnet.pth")
    # writer.close()
