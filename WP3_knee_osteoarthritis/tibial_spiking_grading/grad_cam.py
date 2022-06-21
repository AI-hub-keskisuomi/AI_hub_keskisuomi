from __future__ import print_function, division

import cnn_model
import dataset
import test

import os
import random
import json
import argparse

import numpy as np

import matplotlib.pyplot as plt

import torch

from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
import torchcam.methods

from torchcam.utils import overlay_mask

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


def create_dir(path):

    try:
        os.mkdir(path)
    except FileExistsError:
        print("Directory {} exists already".format(path))
    except:
        raise Exception(
            "Creation of the directory {} failed for some reason".format(path)
        )


def export_gradcam_figs(
    model, test_results, dir_suffix, image_mean, image_std, exp_dir, device
):

    # create directory for gradcam plots

    dir_name = "grad_cam_{}".format(dir_suffix)

    plot_path = os.path.join(exp_dir, dir_name)

    # create root dir
    create_dir(plot_path)

    # create subdirs for correct and misclassified
    subdir_dict = {
        subdir: os.path.join(plot_path, subdir)
        for subdir in ["correct", "misclassified"]
    }

    for key, path in subdir_dict.items():
        # create subdir
        create_dir(path)

        # process gradcam visualizations

        for index, input_tensor in enumerate(test_results[key]["images"]):

            # deprocess the image for visualization
            img = input_tensor.numpy().transpose((1, 2, 0))
            mean = np.array(image_mean)
            std = np.array(image_std)
            img = img * std + mean
            img = np.clip(img, 0, 1)
            # Rescaling grey scale between 0-255
            img = (np.maximum(img, 0) / img.max()) * 255.0
            # Convert to uint
            img = np.uint8(img)

            # eval mode
            model.eval()

            # config the cam
            cam_extractor = torchcam.methods.SmoothGradCAMpp(
                model, target_layer=model.last_conv("resnext")
            )

            # Preprocess your data (turn into 4D tensor with batch size of 1) and feed it to the model
            out = model(input_tensor.unsqueeze(0).to(device))

            # Retrieve the CAM by passing the class index (argmax) and the model output
            activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)

            # Resize the CAM and overlay it
            result = overlay_mask(
                to_pil_image(img),
                to_pil_image(activation_map[0].squeeze(0), mode="F"),
                alpha=0.5,
            )

            plt.subplot(1, 2, 1)
            plt.imshow(img)
            gd_title = "Ground truth: {}".format(test_results[key]["labels"][index])
            plt.title(gd_title)
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.imshow(result)
            res_title = "Prediction: {}".format(test_results[key]["preds"][index])
            plt.title(res_title)
            plt.axis("off")

            plt.tight_layout()

            # save plot
            plt_name = "gradcam_{}".format(index)
            plt.savefig(os.path.join(path, plt_name))


def grad_cam_analysis(data_dir, exp_dir):
    """ Creates grad-cam graphs from the model predictions. Grad-Cams are created for val and test sets.

    Args:
        data_dir (string): dataset directory
        exp_dir (string): experiment directory

    Raises:
        Exception: raises an exeption if model params specify unknown image size
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
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(
                    degrees=(-12, 12), translate=(0.05, 0.01), scale=(0.8, 1.2)
                ),
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
            image_datasets["val"],
            batch_size=batch_size,
            num_workers=4,
            worker_init_fn=seed_worker,
            shuffle=False,
        ),
        "test": torch.utils.data.DataLoader(
            image_datasets["test"], batch_size=batch_size, num_workers=4, shuffle=False
        ),
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("--- train ---")
    freqs(image_datasets["train"])

    print("--- val ---")
    freqs(image_datasets["val"])

    print("--- test ---")
    freqs(image_datasets["test"])

    #### loading the model weights ####

    # init the model
    model = cnn_model.Model(params["model"])

    model.load_state_dict(torch.load(os.path.join(exp_dir, "model_weights.pth")))
    model = model.to(device)

    ### Testing the model ###
    res_val = test.diagnose_model(model, dataloaders["val"], device)
    res_test = test.diagnose_model(model, dataloaders["test"], device)

    ############
    # Grad cam #
    ############

    print("Export gradcams for validation")
    export_gradcam_figs(model, res_val, "val", image_mean, image_std, exp_dir, device)

    print("Export gradcams for test")
    export_gradcam_figs(model, res_test, "test", image_mean, image_std, exp_dir, device)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", default="", help="Experiment directory path")
    args = parser.parse_args()

    #### PARAMS ####

    # data dir
    data_dir = "../datasets/spiking_or_tf_crop_mc2_300x300"

    # experiment source path
    exp_dir = args.s

    # create grad-cam graphs for val and test images
    grad_cam_analysis(data_dir, exp_dir)


if __name__ == "__main__":
    main()
