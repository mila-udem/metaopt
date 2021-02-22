"""
====================
Checkpointing trials
====================

.. hint::

    In short, you should use "{experiment.working_dir}/{trial.hash_params}" to set the path of
    the checkpointing file.

When using multi-fidelity algorithms such as Hyperband it is preferable to checkpoint the trials
to avoid starting training from scratch when resuming a trial. In this tutorial for instance,
hyperband will train VGG11 for 1 epoch, pick the best candidates and train them for 7 more epochs,
doing the same again for 30 epoch, and then 120 epochs. We want to resume training at last epoch
instead of starting from scratch.

Oríon provides a unique hash for trials that can be used to define the unique checkpoint file
path: ``trial.hash_params``. This can be used with the Python API as demonstrated in this example
or with :ref:`commandline_templates`.

With command line
-----------------

The example below is based on the Python API solely. It is also possible to do checkpointing
using the command line API. To this end, your script should accept an argument for the checkpoint
file path. Suppose this argument is ``--checkpoint``, you should call your script with the
following template.

::

    orion hunt -n <exp name>
        ./your_script.sh --checkpoint '{experiment.working_dir}/{trial.hash_params}'

Your script it reponsible to take this checkpoint path, resume from checkpoints or same 
checkpoints.
We will demonstrate below how this can be done with PyTorch, but using Oríon's Python API.

Training code
-------------

We will first go through the training code piece by piece before tackling the hyperparameter
optimization.

First things first, the imports.

"""
import numpy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import SubsetRandomSampler

import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import os
import argparse

#%%
# We will use the data SubsetRandomSampler data loader from PyTorch to split
# the training set into a training and validation sets. We include test set
# here for completeness but won't use it in this example as we only need the training
# data and the validation data for the hyperparameter optimization.
# We use torchvision's transformers to apply the standard transformations on CIFAR10
# images, that is, random cropping, random horizontal flipping and normalization.


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
# Next, we write the function to save checkpoints. It is important to include
# not only the model in the checkpoint, but also the optimizer and the learning rate
# schedule when using one. In this example we will use the exponential learning rate schedule,
# so we checkpoint it. We save the current epoch as well so that we now where we resume from.


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
# To resume from checkpoints, we simply restore the states of the model, optimizer and learning rate
# schedules based on the checkpoint file. If there is no checkpoint path or if the file does not
# exist, we return epoch 1 so that training starts from scratch. Otherwise we return the last
# trained epoch number found in checkpoint file.


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
# Then comes the training loop for one epoch.


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
# Finally the validation loop to compute the validation error rate.


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

    return 100.0 * (1 - correct / total)


#%%
# We combine all these functions into a main function for the whole training pipeline.
#
# .. note::
#
#     We set ``batch_size`` to 1024 by default, you may need to reduce it depending on your GPU.
#


def main(
    epochs=120,
    learning_rate=0.1,
    momentum=0.9,
    weight_decay=0,
    batch_size=1024,
    gamma=0.97,
    checkpoint=None,
):

    # We create the checkpointing folder if it does not exist.
    if checkpoint and not os.path.isdir(checkpoint):
        os.makedirs(checkpoint)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = models.vgg11()
    model = model.to(device)

    # We define the training criterion, optimizer and learning rate scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)

    # We restore the states of model, optimizer and learning rate scheduler if a checkpoint file is
    # available. This will return the last epoch number of the checkpoint or 1 if no checkpoint.
    start_epoch = resume_from_checkpoint(checkpoint, model, optimizer, lr_scheduler)

    # We build the data loaders. test_loader is here for completeness but won't be used.
    train_loader, valid_loader, test_loader = build_data_loaders(batch_size=batch_size)

    # If no training needed, because the trial was resumed from an epoch equal or greater to number
    # of epochs requested here (``epochs``).
    if start_epoch >= epochs + 1:
        return valid(valid_loader, device, model)

    # Training from last epoch until ``epochs + 1``, checkpointing at end of each epoch.
    for epoch in range(start_epoch, epochs + 1):
        print("epoch", epoch)
        train(train_loader, device, model, optimizer, lr_scheduler, criterion)
        valid_error_rate = valid(valid_loader, device, model)
        save_checkpoint(checkpoint, model, optimizer, lr_scheduler, epoch)

    return valid_error_rate


#%%
# You can test the training pipeline before working with the hyperparameter optimization.


main(epochs=4)


#%%
# HPO code
# --------
#
# We finally implement the hyperparameter optimization loop. We will use Hyperband
# with the number of epochs as the fidelity, using the prior ``fidelity(1, 120, base=4)``.
# Hyperband will thus train VGG11 for 1, 7, 30 and 120 epochs. To explore enough candidates
# at 120 epochs, we set Hyperband with 5 repetitions.
#
# In the optimization loop (``while not experiment.is_done``), we ask Oríon to suggest a new trial
# and then pass the hyperparameter values ``**trial.params`` to ``main()``, specifying the
# checkpoint file with ``f"{experiment.working_dir}/{trial.hash_params}"``.


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
    )

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


#%%
# Let's run the optimization now. You may want to reduce the maximum number of epochs in
# ``fidelity(1, 120, base=4)`` and set the number of ``repetitions`` to 1 to get results more
# quickly. With current configuration, this example takes 2 days to run on a Titan RTX.

experiment = run_hpo()

#%%
# Analysis
# --------
#
# That is all for the checkpointing example. We should nevertheless analyse the results
# before wrapping up this tutorial.
#
# We should first look at the :ref:`sphx_glr_auto_examples_plot_1_regret.py`
# to verify the optimization with Hyperband.

fig = experiment.plot.regret()
fig.show()

#%%
# .. This file is produced by docs/scripts/build_database_and_plots.py
#
# .. raw:: html
#
#     :file: ../_static/hyperband-cifar10_regret.html
#
#
# Moving the cursor over the points, we see that only a handful of trials
# lead to better results with 1 epoch. Otherwise, all other trials with validation error rate
# below 80% were trained for more than 1 epoch.
# The best found result is high, a validation accuracy 23.6%. With VGG11 we could expect to achieve
# lower than 10%. To see if the search space may be the issue, we first look at the
# :ref:`sphx_glr_auto_examples_plot_3_lpi.py`.

fig = experiment.plot.lpi()
fig.show()

#%%
# .. raw:: html
#     :file: ../_static/hyperband-cifar10_lpi.html
#
# The momentum and weight decay had very large priors, yet the different values
# had no important effect on the validation accuracy. We can ignore them.
# For the learning rate and for gamma
# it is worth looking at the :ref:`sphx_glr_auto_examples_plot_4_partial_dependencies.py`
# to see if the search space was perhaps too narrow or too large.

fig = experiment.plot.partial_dependencies(params=["gamma", "learning_rate"])
fig.show()

# sphinx_gallery_thumbnail_path = '_static/restart.png'

#%%
# .. This file is produced by docs/scripts/build_database_and_plots.py
#
# .. raw:: html
#     :file: ../_static/hyperband-cifar10_partial_dependencies_params.html
#
# The main culprit for the high validation error rate seems to be the wide prior for ``gamma``.
# Because of this Hyperband spent most of the computation time on bad ``gamma``s. This prior
# should be narrowed to ``uniform(0.995, 1)``.
# The prior for the learning rate could also be narrowed to ``loguniform(0.001, 0.1)`` to
# help the optimization.
#
# Note that Hyperband could also find better results without adjusting the search space, but
# it would required significantly more repetitions.
