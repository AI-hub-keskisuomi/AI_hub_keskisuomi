from __future__ import print_function, division

import cnn_model
import dataset
import train
import test

import copy
import os
import random
import json
import csv
import argparse

import numpy as np

from sklearn.model_selection import ParameterGrid

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from torchvision import transforms


# https://pytorch.org/docs/stable/notes/randomness.html
# https://discuss.pytorch.org/t/reproducibility-with-all-the-bells-and-whistles/81097


def seed_all(seed):
    if not seed:
        seed = 10

    print("[ Using Seed : ", seed, " ]")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def search_hyperparams(params, data_dir, csv_file, device):
    """Search optimal hyperparameters using gridsearch.
    Models are selected using the validation accuracy.

    Args:
        params (dictionary): dictionary of parameters to search
        data_dir (string): dataset directory
        csv_file (string): path for results.csv file
        device (string): define the device to use (cpu or cuda)

    Raises:
        Exception: Raises an exception if image size defined in param dictionary is not 224 or 300
        Exception: Raises an exception if optimizer defined in param dictionary is not adam or SGD

    Returns:
        model weights, dictionary: top model weigths and its hyperparameters parameters
    """

    print("\n---Starting param search---")

    #### start param search ####
    # initialize best acc and params
    best_acc = 0.0
    best_params = {}

    # create param iterator
    param_grid = ParameterGrid(params)
    params_total = len(param_grid)
    params_current = 1

    # hack for getting the ordering of the params right
    # headers is used in results.csv
    for p in param_grid:
        headers = list(p.keys())
        break

    # append headers for train and validation accuracy
    headers.append("train_acc")
    headers.append("val_acc")

    # write header row to a new csv file
    with open(csv_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

    for param in param_grid:

        # set seed, this need to be reset for each train eval loop to get reproducible results
        seed_all(10)

        print("\n------------------------")
        print("Evaluating params {c}/{t} :".format(c=params_current, t=params_total))
        print(param)
        print("------------------------")

        ### img size and augmentations

        # image mean and std per channel calculated from train data
        image_mean = [0.7095, 0.7095, 0.7095]

        # for use different image std for different img_sizes
        if param["img_size"] == 300:
            img_size = 300
            image_std = [0.1620, 0.1620, 0.1620]
            # images are already 300x300 so resize is not needed
            train_t = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomAffine(
                        degrees=(-12, 12), translate=(0.05, 0.01), scale=(0.8, 1.2)
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=image_mean, std=image_std),
                ]
            )
            val_t = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=image_mean, std=image_std),
                ]
            )

        elif param["img_size"] == 224:
            img_size = 224
            image_std = [0.1570, 0.1570, 0.1570]

            train_t = transforms.Compose(
                [
                    transforms.Resize((img_size, img_size)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomAffine(
                        degrees=(-12, 12), translate=(0.05, 0.01), scale=(0.8, 1.2)
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=image_mean, std=image_std),
                ]
            )

            val_t = transforms.Compose(
                [
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=image_mean, std=image_std),
                ]
            )
        else:
            raise Exception("Not recognised img_size")

        image_datasets = {
            "train": dataset.SpikingDataset(
                csv_file=os.path.join(data_dir, "train/dataset.csv"),
                root_dir=os.path.join(data_dir, "train"),
                transform=train_t,
            ),
            "val": dataset.SpikingDataset(
                csv_file=os.path.join(data_dir, "val/dataset.csv"),
                root_dir=os.path.join(data_dir, "val"),
                transform=val_t,
            ),
            "test": dataset.SpikingDataset(
                csv_file=os.path.join(data_dir, "test/dataset.csv"),
                root_dir=os.path.join(data_dir, "test"),
                transform=val_t,
            ),
        }

        batch_size = param["batch_size"]

        dataloaders = {
            "train": torch.utils.data.DataLoader(
                image_datasets["train"],
                batch_size=batch_size,
                num_workers=4,
                worker_init_fn=seed_worker,
                shuffle=True,
            ),
            "val": torch.utils.data.DataLoader(
                image_datasets["val"],
                batch_size=batch_size,
                num_workers=4,
                worker_init_fn=seed_worker,
                shuffle=False,
            ),
            "test": torch.utils.data.DataLoader(
                image_datasets["test"],
                batch_size=batch_size,
                num_workers=4,
                shuffle=False,
            ),
        }

        ####################
        # initialize model #
        ####################
        model = cnn_model.Model(param["model"])
        model = model.to(device)

        ################
        # model params #
        ################

        epochs = param["num_epochs"]
        criterion = nn.CrossEntropyLoss()

        if param["optimizer"] == "sgd":
            optimizer = optim.SGD(
                model.parameters(), lr=param["learning_rate"], momentum=0.9
            )
        elif param["optimizer"] == "adam":
            optimizer = optim.Adam(model.parameters(), lr=param["learning_rate"])
        else:
            raise Exception("Error: optimizer was not either sgd or adam")

        # Decay LR by a factor of <gamma> every <step_size> epochs
        exp_lr_scheduler = lr_scheduler.StepLR(
            optimizer, step_size=param["step_size"], gamma=param["gamma"]
        )

        #### Train model ####
        model = train.train_model(
            model,
            dataloaders,
            image_datasets,
            criterion,
            optimizer,
            exp_lr_scheduler,
            device,
            num_epochs=epochs,
        )
        ####################

        #### Test model ####
        res_train = test.test_model(model, dataloaders["train"], device)
        res_val = test.test_model(model, dataloaders["val"], device)

        train_acc = res_train["correct_samples"] / res_train["total_samples"]
        val_acc = res_val["correct_samples"] / res_val["total_samples"]
        ####################

        # write results to a file
        results_row = list(param.values())
        results_row.append(train_acc)
        results_row.append(val_acc)

        with open(csv_file, "a") as f:
            writer = csv.writer(f)
            writer.writerow(results_row)

        params_current += 1

        if val_acc > best_acc:
            print("\nFound a new top model!")
            best_acc = val_acc
            best_params = param
            best_model_wts = copy.deepcopy(model.state_dict())

    ### end of search ###

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, best_params


############################################


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", default="", help="Experiment directory path")
    parser.add_argument("--gpu", default=0, help="Index for which GPU to use")
    args = parser.parse_args()

    # experiment source path
    exp_dir = args.s

    results_file = os.path.join(exp_dir, "results.csv")

    # dataset dir
    data_dir = "../datasets/spiking_or_tf_crop_mc2_300x300"

    # use cuda if possible
    cuda_x = "cuda:{}".format(args.gpu)
    device = torch.device(cuda_x if torch.cuda.is_available() else "cpu")

    #### Load params ####
    json_path = os.path.join(exp_dir, "params.json")
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(
        json_path
    )
    with open(json_path) as f:
        params = json.load(f)

    #### start hyperparameter search
    model, model_params = search_hyperparams(params, data_dir, results_file, device)

    print("\nSearch over, saving results..")

    #### saving the model weights ####
    torch.save(model.state_dict(), os.path.join(exp_dir, "model_weights.pth"))

    #### saving model parameters ####
    with open(os.path.join(exp_dir, "model_params.json"), "w") as fp:
        json.dump(model_params, fp, indent=4)


if __name__ == "__main__":
    main()
