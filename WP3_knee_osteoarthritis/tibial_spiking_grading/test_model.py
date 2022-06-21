from __future__ import print_function, division

import cnn_model
import dataset
import test

import os
import random
import json
import argparse

import numpy as np

from sklearn.metrics import confusion_matrix, recall_score, precision_score

import torch
import torch.nn as nn

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


def freqs(dataset):
    freqs = torch.as_tensor(dataset.targets).bincount()
    print("Total images {}".format(freqs.sum()))
    classes = dataset.classes
    for idx, class_name in enumerate(classes):
        print("Num of {} images: {}".format(class_name, freqs[idx]))


def binary_sensitivity_specificity(true, pred):
    """
    Returns Sensitivity and Specificity for binary classification.

    Returns: sensitivity, specificity
    """

    # calculate class specific recall scores
    recall_classes = recall_score(true, pred, average=None)

    # recall for positive class (class 1) is same as sensitivity
    # recall for negative class (class 0) is same as specificity
    return recall_classes[1], recall_classes[0]


def test_model(data_dir, exp_dir):
    """Test the model

    Args:
        data_dir (string): dataset directory
        exp_dir (string): experiment directory

    Raises:
        Exception: Image size specified in parameters is not 224 or 300.
    """

    # seed the run
    seed_all(10)

    # load params
    json_path = os.path.join(exp_dir, "model_params.json")
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(
        json_path
    )
    with open(json_path) as f:
        params = json.load(f)

    print("Testing model with params")
    print(params)

    ### img size and augmentations

    # image mean and std per channel calculated from train data
    image_mean = [0.7095, 0.7095, 0.7095]

    # for use different image std for different img_sizes
    if params["img_size"] == 300:
        img_size = 300
        image_std = [0.1620, 0.1620, 0.1620]
        # images are already 300x300 so resize is not needed
        val_t = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=image_mean, std=image_std),
            ]
        )

    elif params["img_size"] == 224:
        img_size = 224
        image_std = [0.1570, 0.1570, 0.1570]

        val_t = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=image_mean, std=image_std),
            ]
        )
    else:
        raise Exception("Not recognised img_size")

    batch_size = params["batch_size"]

    image_datasets = {
        "train": dataset.SpikingDataset(
            csv_file=os.path.join(data_dir, "train/dataset.csv"),
            root_dir=os.path.join(data_dir, "train"),
            transform=val_t,
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

    dataloaders = {
        "train": torch.utils.data.DataLoader(
            image_datasets["train"],
            batch_size=batch_size,
            num_workers=4,
            worker_init_fn=seed_worker,
            shuffle=True,
        ),
        "val": torch.utils.data.DataLoader(
            image_datasets["val"], batch_size=batch_size, num_workers=4, shuffle=False
        ),
        "test": torch.utils.data.DataLoader(
            image_datasets["test"], batch_size=batch_size, num_workers=4, shuffle=False
        ),
    }

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    print("--- train ---")
    freqs(image_datasets["train"])

    print("--- val ---")
    freqs(image_datasets["val"])

    print("--- test ---")
    freqs(image_datasets["test"])

    #### loading the model weights ####

    # init the model
    loaded_model = cnn_model.Model(params["model"])

    loaded_model.load_state_dict(torch.load(os.path.join(exp_dir, "model_weights.pth")))
    loaded_model = loaded_model.to(device)

    criterion = nn.CrossEntropyLoss()

    ### Testing the model ###

    res_train = test.test_model(loaded_model, dataloaders["train"], device, criterion)
    res_val = test.test_model(loaded_model, dataloaders["val"], device, criterion)
    res_test = test.test_model(loaded_model, dataloaders["test"], device, criterion)

    train_acc = res_train["correct_samples"] / res_train["total_samples"]
    val_acc = res_val["correct_samples"] / res_val["total_samples"]
    test_acc = res_test["correct_samples"] / res_test["total_samples"]

    print("\n-------")
    print("Train acc:", train_acc)
    print("Val acc:", val_acc)
    print("Test acc:", test_acc)
    print("-------")

    # sensitivity and specificity

    train_sensitivity, train_specificity = binary_sensitivity_specificity(
        res_train["true_labels"], res_train["pred_labels"]
    )
    val_sensitivity, val_specificity = binary_sensitivity_specificity(
        res_val["true_labels"], res_val["pred_labels"]
    )
    test_sensitivity, test_specificity = binary_sensitivity_specificity(
        res_test["true_labels"], res_test["pred_labels"]
    )

    print("\n-------")
    print("Train sensitivity:", train_sensitivity)
    print("Train specificity", train_specificity)
    print("\nValidation sensitivity:", val_sensitivity)
    print("Validation specificity", val_specificity)
    print("\nTest sensitivity:", test_sensitivity)
    print("Test specificity", test_specificity)
    print("-------")

    train_precision = precision_score(
        res_train["true_labels"], res_train["pred_labels"], pos_label=1
    )
    val_precision = precision_score(
        res_val["true_labels"], res_val["pred_labels"], pos_label=1
    )
    test_precision = precision_score(
        res_test["true_labels"], res_test["pred_labels"], pos_label=1
    )

    print("\n-------")
    print("Train precision", train_precision)
    print("Validation precision", val_precision)
    print("Test precision", test_precision)
    print("-------")

    cf_matrix_train = confusion_matrix(
        res_train["true_labels"], res_train["pred_labels"]
    )
    print("Confusion matrix for train data")
    print(cf_matrix_train)

    cf_matrix_val = confusion_matrix(res_val["true_labels"], res_val["pred_labels"])
    print("\nConfusion matrix for validation data")
    print(cf_matrix_val)

    cf_matrix_test = confusion_matrix(res_test["true_labels"], res_test["pred_labels"])
    print("\nConfusion matrix for test data")
    print(cf_matrix_test)

    print("\nDataset losses")
    print("Train", res_train["loss"])
    print("Val", res_val["loss"])
    print("Test", res_test["loss"])


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", default="", help="Experiment directory path")
    args = parser.parse_args()

    #### PARAMS ####

    # data dir
    data_dir = "../datasets/spiking_or_tf_crop_mc2_300x300"

    # experiment source path
    exp_dir = args.s

    test_model(data_dir, exp_dir)


if __name__ == "__main__":
    main()
