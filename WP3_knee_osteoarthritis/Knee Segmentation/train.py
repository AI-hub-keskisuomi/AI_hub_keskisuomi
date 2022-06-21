import argparse
import sys
from pathlib import Path
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from utils.data_loading import BasicDataset
from utils.metrics import dice_loss
from utils.utils import average_meter
from evaluate import evaluate
from unet import UNet
from drn.drn import DRNSeg
import numpy as np

def train_net(net,
              device,
              data_directory,
              epochs: int = 5,
              batch_size: int = 1,
              learning_rate: float = 0.001,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              img_scale: float = 1,
              amp: bool = False,
              name: str = "unet"):
              
    dir_img=os.path.join(data_directory,"data")
    dir_mask=os.path.join(data_directory,"target")
    dir_checkpoint="./checkpoints"
    dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # 2. Split into train / validation partitions
   
    #in a way that both knees of a given participant are in the same set avoiding a possible  source of leakage
    patient_ids=list(set([s[:-1] for s in dataset.ids]))
    n_val = int(len(patient_ids) * val_percent)
    n_train = len(patient_ids) - n_val
    train_set, val_set = random_split(patient_ids, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    train_idxs=[]
    val_idxs=[]
    for i, id in enumerate(dataset.ids):
        if id[:-1] in val_set:
            val_idxs.append(i)
        else:
            train_idxs.append(i)
    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=1)
    n_train,n_val=len(train_idxs),len(val_idxs)
    train_loader = DataLoader(torch.utils.data.Subset(dataset,train_idxs), shuffle=False, **loader_args)
    val_loader = DataLoader(torch.utils.data.Subset(dataset,val_idxs), shuffle=False, drop_last=True, **loader_args)

    
    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if net.n_classes>1 else nn.BCELoss()
    global_step = 0
    running_avg=average_meter()
    
    # 5. Begin training
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image']
                true_masks = batch['mask']
                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)
                
                if net.n_classes==1:
                    true_masks= true_masks[np.newaxis, ...].float()/255
                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = net(images)
                    loss=criterion(torch.sigmoid(masks_pred), true_masks)
                    if net.n_classes>1:
                        onehot=F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float() 
                        loss+=dice_loss(F.softmax(masks_pred, dim=1).float(),
                                            onehot, multiclass=net.n_classes>1) 

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                running_avg.update(loss.item())
                pbar.set_postfix(**{'Avg loss': running_avg.avg})

                # Evaluation round
                division_step = (n_train // (10 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:

                        val_score = evaluate(net, val_loader, device)
                        scheduler.step(val_score)


        if save_checkpoint and (epoch+1)%5==0:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), os.path.join(dir_checkpoint, 'checkpoint_epoch_{}_{}.pth'.format(name,epoch + 1)))


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--architecture','-a', choices=['unet', 'drn'],default='unet',help='Architecture to train, possible choices: unet, drn')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--input-dir', '-i', metavar='I',dest='input_dir', type=str, default='./training_data/', help='Directory with training data divided into folders named data and target')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--n-classes', '-nc', dest='n_classes', metavar='NC', type=int, default=1, help='Number of classes')
    parser.add_argument('--n-channels', '-nch', dest='n_channels', metavar='NCH', type=int, default=1, help='Number of channels in input images')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=0.00001,
                        help='Learning rate', dest='lr')
    parser.add_argument('--scale', '-s', type=float, default=1, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--gpu',default=0, help='Index of the cuda device to use')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(device)
    if args.architecture=='unet':
        net = UNet(n_channels=args.n_channels, n_classes=args.n_classes, bilinear=True)
    else:
        net=DRNSeg(model_name="drn_d_105",n_channels=args.n_channels,n_classes=args.n_classes)

    net.to(device=device)
    try:
        train_net(net=net,
                   data_directory=args.input_dir,
                   epochs=args.epochs,
                   batch_size=args.batch_size,
                   learning_rate=args.lr,
                   device=device,
                   img_scale=args.scale,
                   val_percent=args.val / 100,
                   amp=args.amp,
                   name=args.architecture)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        sys.exit(0)
