#FLIP ORIENTATION 1
import glob
import os

import numpy as np
from skimage.io import imsave , imread

path=r'C:\Users\DeepWorkspace\Desktop\KL Data\kneeKL299\GANFINAL\0_1/'
os.chdir(path)
files = glob.glob('*L.png')

path3 = r'C:\Users\DeepWorkspace\Desktop\KL Data\kneeKL299\GANFINAL\0_1\Flip/'
os.mkdir(path3)

for item in files:
    image = imread ( path + item )
    image = np.fliplr(image)
    imsave (os.path.join ( path3 , item ) ,image)
