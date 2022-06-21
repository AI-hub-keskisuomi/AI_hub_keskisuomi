import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import os
from utils.data_loading import BasicDataset
from torch.utils.data import DataLoader
from unet import UNet
from drn.drn import DRNSeg


from utils.metrics import multiclass_dice_coeff, dice_coeff, IOU

def evaluate(net, dataloader, device, metric="dice", threshold=.5):
    net.eval()
    num_val_batches = len(dataloader)
    score = 0

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        
        if net.n_classes==1:
            mask_true= mask_true[np.newaxis, ...].float()
            
        mask_true = mask_true if net.n_classes==1 else F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)

            # convert to one-hot format
            if net.n_classes == 1:
                mask_pred = (torch.sigmoid(mask_pred) > threshold).float()
                # compute the Dice score
                if metric=="dice":
                    score+=dice_coeff(mask_pred,mask_true,reduce_batch_first=False)
                else:
                    iou= IOU(mask_pred, mask_true)
                    score +=iou
            else:
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)

    net.train()
    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return score
    return score / num_val_batches
def get_args():
    parser = argparse.ArgumentParser(description = 'Evaluate the given segmentation-model using the IOU-metric')
    parser.add_argument('--architecture', '-a', choices = ['unet', 'drn'], default = 'unet', help = 'Architecture to evaluate, current choices: unet, drn')
    parser.add_argument('--model', '-m', default = 'MODEL.pth', metavar = 'FILE', 
                        help = 'Specify the file in which the model is stored')
    parser.add_argument('--input-dir', '-i', metavar = 'I', dest = 'input_dir', type = str, default = 'test_data', help = 'Directory with test data divided into folders named data and target')
    parser.add_argument('--n-classes', '-nc', dest = 'n_classes', metavar = 'NC', type = int, default = 1, help = 'Number of classes')
    parser.add_argument('--n-channels', '-nch', dest = 'n_channels', metavar = 'NCH', type = int, default = 1, help = 'Number of channels in input images')
    parser.add_argument('--scale', '-s', type = float, default = 1., help = 'Downscaling factor of the images')
    parser.add_argument('--mask-threshold', '-t', type = float, default = 0.5,  help = 'Minimum probability value to consider a mask pixel white')
    return parser.parse_args()

if __name__=="__main__":
    args=get_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset=BasicDataset(os.path.join(args.input_dir,"data"),os.path.join(args.input_dir,"target"),augment=False,scale=args.scale)
    dataloader=DataLoader(dataset)
    if args.architecture == 'unet':
        model = UNet(n_channels = args.n_channels, n_classes = args.n_classes, bilinear = True)
    else:
        model = DRNSeg(model_name = "drn_d_105", n_channels = args.n_channels, n_classes = args.n_classes)

    model.load_state_dict(torch.load(args.model, map_location = device))
    model.to(device=torch.device(device))
    print(evaluate(model,dataloader,device,metric="not dice"))