import torch
import numpy as np
from torch import Tensor


def dice_coeff(prediction: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert prediction.size() == target.size()
    if prediction.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {prediction.shape})')

    if prediction.dim() == 2 or reduce_batch_first:
        inter = torch.dot(prediction.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(prediction) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(prediction.shape[0]):
            dice += dice_coeff(prediction[i, ...], target[i, ...])
        return dice / prediction.shape[0]

def IOU(prediction: Tensor, target: Tensor):
    IOU=np.logical_and(prediction.cpu().numpy().round().astype("uint8"),\
                                 target.cpu().numpy().round().astype("uint8")).sum()
    IOU=IOU/np.logical_or(prediction.cpu().numpy().round().astype("uint8"),\
                                     target.cpu().numpy().round().astype("uint8")).sum()    
    return IOU


def multiclass_dice_coeff(prediction: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert prediction.size() == target.size()
    dice = 0
    for channel in range(prediction.shape[1]):
        dice += dice_coeff(prediction[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return dice / prediction.shape[1]


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)
