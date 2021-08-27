import random
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

import numpy as np
from sklearn.metrics import accuracy_score


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,  out_channels=32, kernel_size=3, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=0)
        self.dropout1 = nn.Dropout(p=0.25)
        self.linear1  = nn.Linear(in_features=12 * 12 * 64, out_features=128)
        self.dropout2 = nn.Dropout(p=0.50)
        self.linear2  = nn.Linear(in_features=128, out_features=10)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = F.relu(self.conv1(X))
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2)
        X = torch.flatten(X, start_dim=1)
        X = self.dropout1(X)
        X = F.relu(self.linear1(X))
        X = self.dropout2(X)
        X = self.linear2(X)
        X = F.log_softmax(X, dim=1)
        return X


class ScriptModel(torch.nn.Module):

    def __init__(self, model: Model):
        super().__init__()
        self.model = model

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.model(X)
        X = torch.exp(X)
        return X


def main():
    # random seed
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # load data
    transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])

    train_dataset = MNIST(root='./data', train=True,
                          download=True, transform=transforms)

    valid_dataset = MNIST(root='./data', train=False,
                          download=True, transform=transforms)

    # prepare dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=64, num_workers=8)
    valid_dataloader = DataLoader(valid_dataset, batch_size=64, num_workers=8)

    # init model and loss
    model = Model()
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # train model
    trange = tqdm.tqdm(train_dataloader)

    for batch, targets in trange:
        optimizer.zero_grad()
        model_out = model(batch)
        loss = criterion(model_out, targets)
        loss.backward()
        optimizer.step()

        status = f'Loss: {loss.item():.8f}'
        trange.set_description(status, refresh=True)

    # valid model
    with torch.no_grad():
        model.eval()
        trange = tqdm.tqdm(valid_dataloader)
        labels = []
        predictions = []

        for batch, targets in trange:
            model_out = model(batch)
            model_out = torch.exp(model_out)
            digits = torch.argmax(model_out, dim=1)
            predictions.extend(digits.tolist())
            labels.extend(targets.tolist())

        print('accuracy:', accuracy_score(labels, predictions))

    # trace model
    script_model = ScriptModel(model)
    script_model.eval()

    inputs = torch.randn(1, 1, 28, 28)
    jit_model = torch.jit.trace(script_model, inputs)
    jit_model.save('model.pt')


if __name__ == '__main__':
    main()

