"""Reproduce best MLP run from assignment 1 (20260315-192955)
Writes logs to 3-hypertuning-rnn/modellogs and passes scheduler=None to Trainer.
"""
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from mltrainer import Trainer, TrainerSettings, ReportTypes
from mltrainer import metrics
from mltrainer.preprocessors import BasePreprocessor
from mads_datasets import DatasetFactoryProvider, DatasetType


class SimpleMLP(nn.Module):
    def __init__(self, units1=256, units2=128, num_classes=10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, units1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(units1, units2)
        self.fc3 = nn.Linear(units2, num_classes)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


def main():
    # moved into 3-hypertuning-rnn/scripts; go up two levels to project root
    root = Path(__file__).resolve().parents[2]
    logdir = root / '3-hypertuning-rnn' / 'modellogs'
    logdir.mkdir(parents=True, exist_ok=True)

    # dataset
    factory = DatasetFactoryProvider.create_factory(DatasetType.FASHION)
    pre = BasePreprocessor()
    streamers = factory.create_datastreamer(batchsize=128, preprocessor=pre)
    train = streamers['train'].stream()
    valid = streamers['valid'].stream()

    # config taken from assignment1 best run
    units1 = 256
    units2 = 128
    epochs = 10
    lr = 1e-3
    weight_decay = 1e-5
    train_steps = 937
    valid_steps = 156

    model = SimpleMLP(units1=units1, units2=units2)
    accuracy = metrics.Accuracy()
    loss_fn = nn.CrossEntropyLoss()

    settings = TrainerSettings(
        epochs=epochs,
        metrics=[accuracy],
        logdir=str(logdir),
        train_steps=train_steps,
        valid_steps=valid_steps,
        reporttypes=[ReportTypes.TENSORBOARD, ReportTypes.TOML],
    )

    trainer = Trainer(
        model=model,
        settings=settings,
        loss_fn=loss_fn,
        optimizer=optim.Adam,
        traindataloader=train,
        validdataloader=valid,
        scheduler=None,
    )

    print(datetime.now(), 'Starting reproduction of best MLP run')
    try:
        trainer.loop()
        print(datetime.now(), 'Finished reproduction run')
    except Exception as e:
        print('Reproduction failed:', e)


if __name__ == '__main__':
    main()
