import os

import pandas as pd
import numpy as np

from PIL import Image

import torch
from torch.utils.data import Dataset


class SpikingDataset(Dataset):
    """Tibial spiking dataset"""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.annotations_frame = pd.read_csv(csv_file)

        self.root_dir = root_dir
        self.transform = transform
        self.classes = [0, 1]
        self.annotations_frame["spiking or"] = (
            np.logical_or(
                self.annotations_frame["Med. spiking"],
                self.annotations_frame["Lat. spiking"],
            )
            * 1
        )
        self.targets = self.annotations_frame["spiking or"].to_numpy()

    def __len__(self):
        return len(self.annotations_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.annotations_frame.iloc[idx, :]["img"]
        spiking_grade = self.annotations_frame.iloc[idx, :]["spiking or"]

        # form a path to the image file
        # file is located in /root/<spiking grade>/<filename>
        img_path = os.path.join(self.root_dir, str(spiking_grade), img_name)

        image = Image.open(img_path)
        image = image.convert("RGB")

        target = np.array(spiking_grade)
        target = target.astype(int)

        if self.transform:
            image = self.transform(image)

        sample = {"image": image, "target": target}

        return sample
