import cv2 as cv
from os import listdir
from os.path import splitext
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import utils.data_transforms as transforms

EPS=1e-3
class BasicDataset(Dataset):    
    def __init__(self, images_dir: str, masks_dir: str, scale: float = 1.0, augment: bool = False, mask_suffix: str = '',grayscale: bool=True):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.grayscale=grayscale

        self.augment=augment
        t = [transforms.RandomSaturation()]
        self.transforms= transforms.Compose(t)

        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir},'+
                                'make sure you put your images there')
    
    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale, is_mask):
        w, h = pil_img.size
        new_w, new_h = int(scale * w), int(scale * h)
        assert new_w > 0 and new_h > 0, 'Scale is too small, resized images would have no pixel'
        if scale<=1:
            pil_img = pil_img.resize((new_w, new_h),\
                 resample=Image.NEAREST if is_mask else Image.BICUBIC)
        if is_mask:
            pil_img=pil_img.convert("L")
        img_ndarray = np.asarray(pil_img)

        if img_ndarray.ndim == 2 and not is_mask:
            img_ndarray=cv.equalizeHist(img_ndarray.astype("uint8"))
            img_ndarray = img_ndarray[np.newaxis, ...]
            #img_ndarray=img_ndarray-img_ndarray.min()
            #img_ndarray =img_ndarray/img_ndarray.max()
        elif not is_mask:
            img_ndarray = img_ndarray.transpose((2, 0, 1))
        return img_ndarray

    @classmethod
    def load(cls, filename,grayscale=True):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        elif grayscale:
            return Image.open(filename).convert("L")
        else:
            return Image.open(filename).convert('RGB')

    def __getitem__(self, idx):
        if isinstance(idx,str):
            name=idx
        elif isinstance(idx,int):
            name = self.ids[idx]
        elif isinstance(idx,list):
            return[self[ix] for ix in idx]
        mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        assert len(img_file) == 1, \
             f'Either no image or multiple images found for the ID {name}: {img_file}'
        mask = self.load(mask_file[0], grayscale=True)
        img = self.load(img_file[0], grayscale=self.grayscale)

        assert img.size == mask.size, \
            'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'
        if self.augment: 
            img, mask=self.transforms(img,mask)

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)
        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.round(EPS+torch.as_tensor(mask.copy())).long().contiguous()
        }