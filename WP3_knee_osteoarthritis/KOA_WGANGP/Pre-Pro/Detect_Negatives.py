#%% DETECT NEGATIVES AND FLIP 2
import os

import numpy as np
from skimage import exposure
from skimage.io import imsave , imread
from skimage.util import invert

path=r'C:\Users\DeepWorkspace\Desktop\KL Data\kneeKL299\GANFINAL\0_1/'
path2 = path+str('/negs')
os.mkdir(path2)

count=0
for f in os.listdir(path):
    name, ext = os.path.splitext(f)
    if ext == '.png':

        image = imread ( os.path.join ( path , f ) )
        image_eq = exposure.equalize_hist(image)
        thresh = 10
        size = len ( image_eq ) - 1
        inv = invert (image_eq)
        merge_norm = np.concatenate ( (image_eq [ size - thresh:size , size - thresh:size ] ,
                                       image_eq [ 0:thresh , size - thresh:size ] ,
                                       image_eq [ 0:thresh , 0:thresh ] ,
                                       image_eq [ size - thresh:size , 0:thresh ]) ).mean ( )

        merge_inv = np.concatenate ( (inv [ size - thresh:size , size - thresh:size ] ,
                                      inv [ 0:thresh , size - thresh:size ] ,
                                      inv [ 0:thresh , 0:thresh ] ,
                                      inv [ size - thresh:size , 0:thresh ]) ).mean ( )

        if merge_norm > merge_inv:
            count += 1
            print ( count )
            imsave ( os.path.join ( path2 , f ) , image )
        else:pass
#%

path = path2
path2 = path+str('/ready')
os.mkdir(path2)

for f in os.listdir(path):
    name, ext = os.path.splitext(f)
    if ext == '.png':

        image = imread ( os.path.join ( path , f ) )
        inv = invert (image)
        imsave ( os.path.join ( path2 , f ) , inv )