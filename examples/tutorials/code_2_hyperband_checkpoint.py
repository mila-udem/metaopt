"""
====================
Checkpointing trials
====================

Long story short, use "{experiment.working_dir}/{trial.hash_params}" to set the path of
the checkpointing file. (now explain why...)

TODO: Need to download data? Or handled by CIFAR10 constructor?


"""
import numpy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import SubsetRandomSampler

import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import os
import argparse

#%%
# Building data loaders...


def build_data_loaders(batch_size, split_seed=1):
    normalize = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]

    augment = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]

    train_set = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transforms.Compose(augment + normalize),
    )
    valid_set = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transforms.Compose(normalize),
    )
    test_set = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transforms.Compose(normalize),
    )

    num_train = 45000
    # num_valid = 5000
    indices = numpy.arange(num_train)
    numpy.random.RandomState(split_seed).shuffle(indices)

    train_idx, valid_idx = indices[:num_train], indices[num_train:]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, sampler=train_sampler, num_workers=5
    )
    valid_loader = torch.utils.data.DataLoader(
        train_set, batch_size=1000, sampler=train_sampler, num_workers=5
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=1000, shuffle=False, num_workers=5
    )

    return train_loader, valid_loader, test_loader


#%%
# Saving checkpoints...


def save_checkpoint(checkpoint, model, optimizer, lr_scheduler, epoch):
    if checkpoint is None:
        return

    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "epoch": epoch,
    }
    torch.save(state, f"{checkpoint}/checkpoint.pth")


#%%
# Resuming checkpoints


def resume_from_checkpoint(checkpoint, model, optimizer, lr_scheduler):
    if checkpoint is None:
        return 1

    try:
        state_dict = torch.load(f"{checkpoint}/checkpoint.pth")
    except FileNotFoundError:
        return 1

    model.load_state_dict(state_dict["model"])
    optimizer.load_state_dict(state_dict["optimizer"])
    lr_scheduler.load_state_dict(state_dict["lr_scheduler"])
    return state_dict["epoch"] + 1  # Start from next epoch


#%%
# Training loop for one epoch.


def train(loader, device, model, optimizer, lr_scheduler, criterion):
    model.train()
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()


#%%
# Compute validation accuracy.


def valid(loader, device, model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return 100.0 * correct / total


#%%
# Writing the whole training pipeline.


def main(
    epochs=120,
    learning_rate=0.1,
    momentum=0.9,
    weight_decay=0,
    batch_size=1024,
    gamma=0.97,
    checkpoint=None,
):

    if checkpoint and not os.path.isdir(checkpoint):
        os.makedirs(checkpoint)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = models.vgg11()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)

    start_epoch = resume_from_checkpoint(checkpoint, model, optimizer, lr_scheduler)

    train_loader, valid_loader, test_loader = build_data_loaders(batch_size=batch_size)

    # If no training needed
    if start_epoch >= epochs + 1:
        valid_acc = valid(valid_loader, device, model)
        return 100 - valid_acc

    for epoch in range(start_epoch, epochs + 1):
        print("epoch", epoch)
        train(train_loader, device, model, optimizer, lr_scheduler, criterion)
        valid_acc = valid(valid_loader, device, model)
        save_checkpoint(checkpoint, model, optimizer, lr_scheduler, epoch)

    return 100 - valid_acc


#%%
# Testing it?


# main(epochs=4)


#%%
# Writing the HPO pipeline

from orion.client import build_experiment


def run_hpo():

    # Specify the database where the experiments are stored. We use a local PickleDB here.
    storage = {
        "type": "legacy",
        "database": {
            "type": "pickleddb",
            "host": "./db.pkl",
        },
    }

    # Load the data for the specified experiment
    experiment = build_experiment(
        "hyperband-cifar10",
        space={
            "epochs": "fidelity(1, 120, base=4)",
            "learning_rate": "loguniform(1e-5, 0.1)",
            "momentum": "uniform(0, 0.9)",
            "weight_decay": "loguniform(1e-10, 1e-2)",
            "gamma": "loguniform(0.97, 1)",
        },
        algorithms={
            "hyperband": {
                "seed": 1,
                "repetitions": 5,
            },
        },
        storage=storage,
        working_dir="./tmp_hyperband_checkpoint",
    )
    print([trial.status for trial in experiment.fetch_trials()])

    trials = 1
    while not experiment.is_done:
        print("trial", trials)
        trial = experiment.suggest()
        if trial is None and experiment.is_done:
            break
        valid_error_rate = main(
            **trial.params, checkpoint=f"{experiment.working_dir}/{trial.hash_params}"
        )
        experiment.observe(trial, valid_error_rate, name="valid_error_rate")
        trials += 1

    print([trial.params["epochs"] for trial in experiment.fetch_trials()])


run_hpo()

#%%
# Bla bla bla
#
# .. This file is produced by docs/scripts/build_database_and_plots.py
#
# .. raw:: html
#     :file: ../_static/hyperband-cifar10_regret.html

#%%
# Bla bla bla
#
# .. This file is produced by docs/scripts/build_database_and_plots.py
#
# .. raw:: html
#     :file: ../_static/hyperband-cifar10_parallel_coordinates.html

#%%
# Bla bla bla
#
# .. This file is produced by docs/scripts/build_database_and_plots.py
#
# .. raw:: html
#     :file: ../_static/hyperband-cifar10_lpi.html

#%%
# Bla bla bla
#
# .. This file is produced by docs/scripts/build_database_and_plots.py
#
# .. raw:: html
#     :file: ../_static/hyperband-cifar10_partial_dependencies.html


# sphinx_gallery_thumbnail_path = '_static/restart.png'
