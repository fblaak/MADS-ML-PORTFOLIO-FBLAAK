#!/usr/bin/env python3
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from mads_datasets import DatasetFactoryProvider, DatasetType
from mltrainer import Trainer, TrainerSettings, ReportTypes, metrics
from mltrainer.preprocessors import BasePreprocessor


def make_streams(batchsize=64):
    factory = DatasetFactoryProvider.create_factory(DatasetType.FASHION)
    pre = BasePreprocessor()
    streamers = factory.create_datastreamer(batchsize=batchsize, preprocessor=pre)
    train_stream = streamers['train'].stream()
    valid_stream = streamers['valid'].stream()
    return streamers, train_stream, valid_stream


class GRUModel(nn.Module):
    def __init__(self, input_size=28, hidden=128, num_layers=1, num_classes=10):
        super().__init__()
        self.rnn = nn.GRU(input_size, hidden, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), 28, 28)
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        return self.fc(out)


class LSTMModel(nn.Module):
    def __init__(self, input_size=28, hidden=128, num_layers=1, num_classes=10):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), 28, 28)
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        return self.fc(out)


class Conv1DRNNModel(nn.Module):
    def __init__(self, input_size=28, hidden=128, num_layers=1, num_classes=10):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=28, out_channels=28, kernel_size=3, padding=1)
        self.rnn = nn.GRU(28, hidden, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), 28, 28)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        return self.fc(out)


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,32,3,padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32,64,3,padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64*7*7,128)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128,10)

    def forward(self,x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


def main():
    # moved into 3-hypertuning-rnn/scripts; go up two levels to project root
    ROOT = Path(__file__).resolve().parents[2]
    LOGDIR = ROOT / '3-hypertuning-rnn' / 'modellogs'
    LOGDIR.mkdir(parents=True, exist_ok=True)

    streamers, train, valid = make_streams(batchsize=64)
    train_steps = len(streamers['train'])
    valid_steps = len(streamers['valid'])

    accuracy = metrics.Accuracy()
    loss_fn = nn.CrossEntropyLoss()

    device = 'cpu'

    # RNN sanity runs
    models = [GRUModel(), LSTMModel(), Conv1DRNNModel()]
    names = ['GRU','LSTM','Conv1D+GRU']
    for m, n in zip(models, names):
        try:
            print(datetime.now(), 'Starting sanity run:', n)
            m.experiment = f'sanity_rnn_{n}'
            settings = TrainerSettings(epochs=1, metrics=[accuracy], logdir=str(LOGDIR), train_steps=50, valid_steps=50, reporttypes=[ReportTypes.TENSORBOARD, ReportTypes.TOML])
            trainer = Trainer(model=m, settings=settings, loss_fn=loss_fn, optimizer=optim.Adam, traindataloader=train, validdataloader=valid, scheduler=None, device=device)
            trainer.loop()
            print(datetime.now(), 'Finished sanity run:', n)
        except Exception as e:
            print('Sanity run failed for', n, e)

    # Conv2D short baseline
    try:
        print(datetime.now(), 'Starting Conv2D short baseline (3 epochs)')
        model = SimpleCNN()
        settings = TrainerSettings(epochs=3, metrics=[accuracy], logdir=str(LOGDIR), train_steps=train_steps, valid_steps=valid_steps, reporttypes=[ReportTypes.TENSORBOARD, ReportTypes.TOML])
        trainer_short = Trainer(model=model, settings=settings, loss_fn=loss_fn, optimizer=optim.Adam, traindataloader=train, validdataloader=valid, scheduler=None, device=device)
        trainer_short.loop()
        print(datetime.now(), 'Finished Conv2D short baseline')
    except Exception as e:
        print('Conv2D short run failed:', e)


if __name__ == '__main__':
    main()
