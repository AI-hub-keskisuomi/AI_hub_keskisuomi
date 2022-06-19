import os

import cv2 as cv
import numpy as np
import pandas as pd

final=[]
gt=[]
maxeslist = [ ]
minslist = [ ]

imageset='C3'
for filename in os.listdir('/Users/fabian/Desktop/QLab/C_ALL/'+str(imageset)):
    img = cv.imread('/Users/fabian/Desktop/QLab/C_ALL/'+str(imageset)+'/'+str(filename),1) # read input
    g_img = img
    zerosb=np.count_nonzero(g_img[ :, :, 0 ] == 0) # read pixels that are zero in channel
    zerosg=np.count_nonzero(g_img[ :, :, 1 ] == 0)
    zerosr=np.count_nonzero(g_img[ :, :, 2 ] == 0)

    if zerosb ==0 and zerosg==0 and zerosr ==0:
        print ('check')
        print (filename) # show if all 3 channels have 0

    gray = cv.cvtColor(g_img, cv.COLOR_BGR2GRAY) #convert grayscale
    (T, threshInv) = cv.threshold(gray, 0, 255,
        cv.THRESH_BINARY_INV | cv.THRESH_OTSU) # create otsu mask
    masked = cv.bitwise_and(img, img, mask=threshInv) # apply mask to image
    masked=masked.astype('float32')

    (T, threshbin) = cv.threshold(gray, 0, 255,
        cv.THRESH_BINARY | cv.THRESH_OTSU) #create reverse mask (for whites to preserve)
    rev_masked = cv.bitwise_and(img, img, mask=threshbin) # apply mask

    rev_masked=rev_masked.astype('float32')
    rev_masked[ rev_masked == 0 ] = 'nan'
    averages = np.nanmean(rev_masked, axis=(0,1)) # the average from the whites
    masked[ :, :, 0 ]/=averages[0]
    masked[ :, :, 1 ]/= averages[1]
    masked[ :, :, 2 ]/= averages[2]

    nonzerob=np.count_nonzero(masked[ :, :, 0 ])
    nonzerog=np.count_nonzero(masked[ :, :, 1 ])
    nonzeror=np.count_nonzero(masked[ :, :, 2 ])
    print(nonzerob,nonzerog,nonzeror)

    spectral_channel = []
    spectral_channel.append(cv.calcHist([masked], [ 0 ], threshInv, [ 256 ],[0,1.4704298]))
    spectral_channel[0]=spectral_channel[0]/(nonzerob+zerosb)

    spectral_channel.append(cv.calcHist([masked], [ 1 ], threshInv, [ 256 ],[0,1.4704298]))
    spectral_channel[1]=spectral_channel[1]/(nonzerog+zerosg)

    spectral_channel.append(cv.calcHist([masked], [ 2 ], threshInv, [ 256 ],[0,1.4704298]))
    spectral_channel[2]=spectral_channel[2]/(nonzeror+zerosr)

    print ('nonzero',nonzeror,'zeorsr', zerosr, 'sumed', (nonzeror+zerosr))
    all_histos = np.concatenate((spectral_channel[ 0 ], spectral_channel[ 1 ], spectral_channel[ 2 ])).T
    
    # all_histos=all_histos/(nonzeropixelcount)
    final.append(all_histos)
    all_histos = []
    print('Data File:', str(filename))
    gt.append(str(filename))

gt = pd.DataFrame(gt)
data = pd.DataFrame(np.concatenate(final))
combined_data = pd.concat([ data, gt ], axis=1)

num_cols = len(list(data)) 
rng = range(1, int(num_cols / 3) + 1)
col_names = [ 'Blue_' + str(i) for i in rng ] + [ 'Green_' + str(i) for i in rng ] + [ 'Red_' + str(i) for i in rng ] + [ 'Class' ]
combined_data.columns = col_names

# combined_data_constantzero = combined_data.loc[:, (combined_data == 0).any(axis=0)]
combined_data = combined_data.loc[:, (combined_data != 0).any(axis=0)]
combined_data.to_csv(r'/Users/fabian/Desktop/QLab/C_ALL/FINAL'+str(imageset)+'A.csv', index=False)
