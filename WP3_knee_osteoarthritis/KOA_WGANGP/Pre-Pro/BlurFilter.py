#%% Bluryness Selection 4

import os
import cv2
from skimage.io import imsave , imread


path=r'C:\Users\DeepWorkspace\Desktop\KL Data\kneeKL299\GANFINAL\3_4/Equalized/'
path2 = r'C:\Users\DeepWorkspace\Desktop\KL Data\kneeKL299\GANFINAL\3_4/Equalized/Clear/'
path3 = r'C:\Users\DeepWorkspace\Desktop\KL Data\kneeKL299\GANFINAL\3_4/Equalized/Blured/'
count = 0
count2 = 0

os.mkdir(path2)
os.mkdir(path3)
blur_thresh=310

for filename in os.listdir ( path ):
    image = imread ( path + filename , as_gray=1 )
    img = cv2.imread ( path + filename)
    v =cv2.Laplacian(img, cv2.CV_64F).var()
    if v >= blur_thresh :
        imsave (os.path.join ( path2 , filename ) ,image)
        count += 1
        print ( count, 'norm')
    else:
        imsave (os.path.join ( path3 , filename ) ,image)
        count += 1
        print ( count, 'blured' )
