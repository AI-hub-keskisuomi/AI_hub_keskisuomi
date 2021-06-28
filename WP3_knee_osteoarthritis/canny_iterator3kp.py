# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 12:03:48 2021

@author: eskaniin
"""



import numpy as np
from numpy.polynomial.polynomial import polyfit, polyval


import matplotlib.pyplot as plt #kuvaajien piirtoon
import time

from scipy.ndimage import gaussian_filter
from skimage import img_as_float
from skimage import exposure
from skimage import draw
from skimage.morphology import disk #luo kiekon muotoisen elementin
from skimage.morphology import skeletonize
from skimage.transform import rescale
from skimage.feature import canny #reunantunnistus
from skimage.filters import sobel_v, sobel_h #reunantunnistus

import pydicom as dicom #dicom-tiedoston lukemiseen
import os #tiedostonkäsittelyyn
# import pickle #tiedostonkäsittelyyn (välitulokset yms.)
# from PIL import Image
import math

from scipy import ndimage #kuvan prosessointifunktioita esim. konvoluutio
from scipy import stats #tilasto- ja todnäk-funktioita 
from scipy import interpolate
# from scipy.interpolate import CubicSpline, UnivariateSpline #splinityökaulu reunoihin
from scipy.spatial import distance, cKDTree #reunapisteiden järjestämiseen
from scipy.ndimage import convolve, label, binary_dilation

from sklearn.model_selection import ParameterGrid #parametrien koontiin



# skaalaa kuvan siten, että pikseli/kuvamillimetri = vakio
def scale(ds, target):
    img = ds.pixel_array
    img = img.astype(np.float64)
    #tämän pitäisi olla oikea, mutta ei toimi kaikissa kuvissa (9249780)
    #http://gdcm.sourceforge.net/wiki/index.php/Imager_Pixel_Spacing
    # factor = float(ds.ImagerPixelSpacing[0])/target 
    factor = float(ds.PixelSpacing[0])/target #tää sen sijaan toimii
    img = rescale(img, factor, multichannel=False)
    return img, ds.PixelSpacing[0]

def scalepng(png, ds, target):
    factor = float(ds.PixelSpacing[0])/target 
    img = rescale(png, factor, multichannel=False)
    return img


# etsii yläkautta luualueet karkeasti
# palauttaa luualueen binäärikuvana ja sen reunojen koordinaatit
# toiminta:
# jakaa kuvan y-akselin mukaan vaakasuikaleisiin
# etsii kirkkaimman alueen, jota reunustaa tarpeeksi vahvat reunat,
# jotka on löydetty Canny-algoritmilla
# alueen täytyy olla tarpeeksi kapea ja riittävän leveä
# alueen hyväksymisessä painotetaan sen etäisyyttä edellisistä reunoista
# parametrit:
#    img = kuva
#    dpos x-akselin suuntainen positiivinen derivaatta (pos. vaakagradientti)
#    dneg x-akselin suuntainen negatiivinen derivaatta (neg. vaakagradientti)
# paluumuuttujat:
#    femur = luurajat kuvamatriisina
#    ilist, jlist = luurajat kuvan indeksilistoina 
def find_femur(img, dpos, dneg):
    # etsitään kirkkain kohokohta, jonka leveys on uskottava
    # alkuosa
    femur = np.zeros(img.shape)
    ihest = 0
    jhest = 1
    ilist = [] #vasemman reunan koordinaattilista
    jlist = [] #oikean reunan koordinaattilista
    for xcoor in range(50, 60):#img.shape[0]):
        dpos_coor = np.where(dpos[xcoor,:])[0]
        dneg_coor = np.where(dneg[xcoor,:])[0]
        highest_intensity = 0
        for i in dpos_coor:
            for j in dneg_coor:
                if (j - i > 30) and (j - i < 100): #luun vähimmäis- ja enim. leveys
                    new_intensity = np.mean(img[xcoor,i:j])
                    if new_intensity > highest_intensity:
                        highest_intensity = new_intensity
                        ihest = i
                        jhest = j
        femur[xcoor, ihest:jhest] = 1
        ilist.append(ihest)
        jlist.append(jhest)
#tää painotus vois riippua etäisyydestä i-iwprev
# ja toisaalta se etäisyys vois vaikuttaa seur. kierroksen iwprevin painoihin            
    #loppuosassa painotetaan edellisten löytöjen perusteella
    for xcoor in range(60, img.shape[0]):
        dpos_coor = np.where(dpos[xcoor,:])[0]
        dneg_coor = np.where(dneg[xcoor,:])[0]
        highest_intensity = 0
        iwprev = np.average(ilist[-5:], weights=[1/5, 1/4, 1/3, 1/2, 1])
        jwprev = np.average(jlist[-5:], weights=[1/5, 1/4, 1/3, 1/2, 1])
        for i in dpos_coor:
            for j in dneg_coor:
                if (j - i > 30) and (j - i < 150): #luun vähimmäis- ja enim. leveys
                    new_intensity = np.mean(img[xcoor,i:j])
                    if (abs(i-iwprev) < 3):
                        new_intensity *= 1.1 #painotetaan vähän
                    if (abs(j-jwprev) < 3):
                        new_intensity *= 1.1 #painotetaan vähän
                    if new_intensity > highest_intensity:
                        highest_intensity = new_intensity
                        ihest = i
                        jhest = j
        femur[xcoor, ihest:jhest] = 1

        ilist.append(ihest)
        jlist.append(jhest)  
        
    return femur, ilist, jlist

# Etsii reisiluun reunapisteen annetusta suorasta. Pisteeksi asetetaan
# ensimmäinen suuntaisderivaattafunktion pohja-arvo, joka on väh.~-a
# parametrit
#    hsob2 = 2. asteen derivaatta (sobelilla)
#    itibiadot = tibian reunapisteiden koordinaatit listana
#    irow = suoran rivikoordinaatit
#    icol = suoran sarakekoordinaatit
#    central_bone_deriv = suoran suuntainen kuvaderivaatta
# paluuarvo
#    femurin reunapiste annetulla suoralla
def find_femur_dot(hsob2, itibiadot, irow, icol, central_bone_deriv):
    # hienosäädetään femurin piste: ensimmäinen pohja-arvo, joka on kuitenkin
    # väh. ~a
    deriv2 = hsob2[irow, icol] #2. asteen suoran suuntainen derivaatta
    sign2 = np.ones(len(deriv2))
    sign2[deriv2 > 0] = 0
    sign_change = np.zeros(len(deriv2))
    #merkataan derivaatan käännekohdat ylös eli (reisi)luun puolelle
    sign_change[1:] = sign2[1:] - sign2[:-1]  #deriv:n käännekohdat (0-kohdat)
    sign_change[sign_change < 0] = 0 #poistetaan väärän suunnan kohdat (turha?)
    sign_change[itibiadot:] = 0 #tibian puoleiset pois
    sign_change[central_bone_deriv > -0.3] = 0 #liian matalat derivaatat pois
    if sum(sign_change) == 0:
        femur_index = 0
    else:
        femur_index = np.nonzero(sign_change)[0][-1]
    return femur_index

# Etsii reisiluun reunapisteen annetusta suorasta. Pisteeksi asetetaan
# ensimmäinen suuntaisderivaattafunktion huippuarvo, joka on väh.~a
def find_tibia_dot(hsob2, ifemurdot, irow, icol, central_bone_deriv):
    # hienosäädetään tibian piste: ensimmäinen huippuarvo, joka on kuitenkin
    # väh. ~a
    deriv2 = hsob2[irow, icol] #2. asteen suoran suuntainen derivaatta
    sign2 = np.ones(len(deriv2))
    sign2[deriv2 < 0] = 0
    sign_change = np.zeros(len(deriv2))
    #merkataan derivaatan käännekohdat alas eli (sääri)luun puolelle
    sign_change[:-1] = sign2[:-1] - sign2[1:] #deriv:n käännekohdat (0-kohdat)
    sign_change[sign_change < 0] = 0 #poistetaan väärän suunnan kohdat (turha?)
    sign_change[:ifemurdot] = 0 #femurin puoleiset pois
    sign_change[central_bone_deriv < 0.3] = 0 #liian pienet derivaatat pois
    if sum(sign_change) == 0:
        tibia_index = len(sign_change) - 1
    else:
        tibia_index = np.nonzero(sign_change)[0][0]
    return tibia_index



# Hakee nivelalueeksi kohdan, jossa pystygradientti on ensin pieni (reisiluun
# reuna) ja sitten suuri (sääriluun reuna)
# Parametrit:
#    img = kuva
#    femur = binäärikuva luualueesta
#    hsob = kuvan pystyderivaatta
#    gdir = gradientin suunta?
def find_joint_space(img, femur, hsob, gdir):
    # sovitetaan luun pitkittäisakselin lävistävä suora
    
    ijoint_space = []
    jsderiv_diff = []
    middledot = np.zeros(img.shape)
    femurdot = np.zeros(img.shape)
    tibiadot = np.zeros(img.shape)
    
    hsob2 = sobel_h(hsob) #2. asteen suuntaisderivaatta
    
    # jalan keskilinja (ja suunta) lineaariregressiolla
    x,y = np.nonzero(femur)
    res = stats.linregress(x, y)
   
    irow = np.arange(femur.shape[0]) #rivinumerot eli -koordinaatit
    offset = 0
    
    global central_bone_deriv 
    # lasketaan keskilinjan suuntaiset gradienttierot
    ireg = np.round(res.intercept + offset + res.slope*irow).astype(int)
    icol = np.minimum([img.shape[1]-1]*img.shape[0], ireg)
    central_bone_deriv = hsob[irow, icol]
    diffmat = np.subtract.outer(central_bone_deriv, central_bone_deriv)
    diffmat = np.tril(diffmat, 0)
    diffmat = np.triu(diffmat, -50) #pitää olla koht lähekkäin
    diffmat[:100,:] = 0
    diffmat[-100:,:] = 0
    diffmat[:,:100] = 0
    diffmat[:,-100:] = 0
    # haetaan suurimman erotuksen sijainnin puoliväli (i(a) - i(b))/2
    imaxdiff = np.unravel_index(np.argmax(diffmat), diffmat.shape)  
    # hienosäädetään tibian piste: ensimmäinen huippuarvo, joka on väh a
    itibia = find_tibia_dot(hsob2, imaxdiff[1], irow, icol, central_bone_deriv)
    itibia = min(itibia, imaxdiff[0])
    # hienosäädetään samoin femurin piste
    ifemur = find_femur_dot(hsob2, itibia, irow, icol, central_bone_deriv)
    ifemur = max(ifemur, imaxdiff[1])
    
    #lasketaan nivelen keskipisteen sijainti
    imiddle = ifemur + 0.5*(itibia - ifemur)
    ijoint_space.append(imiddle)
    imiddle = [imiddle, np.round(res.intercept + res.slope*imiddle)]    
    jsderiv_diff.append(central_bone_deriv[itibia]-central_bone_deriv[ifemur])

    #piirretään pisteet matriisiin
    middledot[int(imiddle[0]), int(imiddle[1])] = 1
    femurdot[ifemur, icol[ifemur]] = 1
    tibiadot[itibia, icol[itibia]] = 1
    miss = 0 #apumuuttuja luun reunan havaitsemiseen
    imaxdiff_mid = imaxdiff
    # Toistetaan nivelen pystysyyntaisen keskipisteen haku luun leveydeltä
    for offset in range(0,-150, -1):
        # icol, imaxdiff, central_bone_deriv = max_grad_diff(img, res, irow, offset)
        ireg = np.round(res.intercept + offset + res.slope*irow).astype(int)
        icol = np.minimum([img.shape[1]-1]*img.shape[0], ireg)
        central_bone_deriv = hsob[irow, icol]
        diffmat = np.subtract.outer(central_bone_deriv, central_bone_deriv)
        diffmat = np.tril(diffmat, 0)
        diffmat = np.triu(diffmat, -50) #pitää olla koht lähekkäin
        diffmat[:100,:] = 0
        diffmat[-100:,:] = 0
        diffmat[:,:100] = 0
        diffmat[:,-100:] = 0
        #haetaan suurimman erotuksen sijainnin puoliväli (i(a) - i(b))/2
        imaxdiff = np.unravel_index(np.argmax(diffmat), diffmat.shape)
        # ifemur = imaxdiff[1]
        # itibia = imaxdiff[0]
        itibia = find_tibia_dot(hsob2, imaxdiff[1], irow, icol, central_bone_deriv)
        itibia = min(itibia, imaxdiff[0])
        ifemur = find_femur_dot(hsob2, itibia, irow, icol, central_bone_deriv)
        ifemur = max(ifemur, imaxdiff[1])
        #keskipiste
        imiddle = ifemur + 0.5*(itibia - ifemur)
        ijsaverage = np.mean(ijoint_space)
        jsddiff = central_bone_deriv[itibia]-central_bone_deriv[ifemur]
        jsddaverage = np.mean(jsderiv_diff)
        #jos gradientin suuntien kulma on liian pieni, ei oteta, koska
        # normaalisti se on n. 180 astetta ja kyljessä 0 astetta
        femur_edge_dir = gdir[ifemur, icol[ifemur]]
        tibia_edge_dir = gdir[itibia, icol[itibia]]
        if (abs(imiddle - ijsaverage) < 20 and 
            abs(jsddiff / jsddaverage) > 0.2 and
            abs(femur_edge_dir - tibia_edge_dir) > 0.5*math.pi): 
            ijoint_space.append(imiddle)
            jsderiv_diff.append(jsddiff)
            imiddle = [imiddle, np.round(res.intercept + offset + res.slope*imiddle)]
            imiddle[1] = min(imiddle[1], img.shape[1]-1)            
            middledot[int(imiddle[0]), int(imiddle[1])] = 1
            femurdot[ifemur, icol[ifemur]] = 1
            tibiadot[itibia, icol[itibia]] = 1
            miss = 0
        else:
            miss += 1
            if miss > 2 and abs(offset) > 30: #polvi ei voi olla minikapea sentään
                break
    
    # Sama toiseen suuntaan
    for offset in range(0,150):
        ireg = np.round(res.intercept + offset + res.slope*irow).astype(int)
        icol = np.minimum([img.shape[1]-1]*img.shape[0], ireg)
        central_bone_deriv = hsob[irow, icol]
        diffmat = np.subtract.outer(central_bone_deriv, central_bone_deriv)
        diffmat = np.tril(diffmat, 0)
        diffmat = np.triu(diffmat, -50) #pitää olla koht lähekkäin
        diffmat[:100,:] = 0
        diffmat[-100:,:] = 0
        diffmat[:,:100] = 0
        diffmat[:,-100:] = 0
        #haetaan suurimman erotuksen sijainnin puoliväli (i(a) - i(b))/2
        imaxdiff = np.unravel_index(np.argmax(diffmat), diffmat.shape)
        # ifemur = imaxdiff[1]
        # itibia = imaxdiff[0]
        itibia = find_tibia_dot(hsob2, imaxdiff[1], irow, icol, central_bone_deriv)
        itibia = min(itibia, imaxdiff[0])
        ifemur = find_femur_dot(hsob2, itibia, irow, icol, central_bone_deriv)
        ifemur = max(ifemur, imaxdiff[1])
        #keskipiste
        imiddle = ifemur + 0.5*(itibia - ifemur)
        ijsaverage = np.mean(ijoint_space)
        jsddiff = central_bone_deriv[itibia]-central_bone_deriv[ifemur]
        jsddaverage = np.mean(jsderiv_diff)
        #jos gradientin suuntien kulma on liian pieni, ei oteta, koska
        # normaalisti se on n. 180 astetta ja kyljessä 0 astetta
        femur_edge_dir = gdir[ifemur, icol[ifemur]]
        tibia_edge_dir = gdir[itibia, icol[itibia]]
        if (abs(imiddle - ijsaverage) < 20 and 
            abs(jsddiff / jsddaverage) > 0.2 and
            abs(femur_edge_dir - tibia_edge_dir) > 0.5*math.pi): 
            ijoint_space.append(imiddle)
            jsderiv_diff.append(jsddiff)
            imiddle = [imiddle, np.round(res.intercept + offset + res.slope*imiddle)]
            imiddle[1] = min(imiddle[1], img.shape[1]-1)            
            middledot[int(imiddle[0]), int(imiddle[1])] = 1
            femurdot[ifemur, icol[ifemur]] = 1
            tibiadot[itibia, icol[itibia]] = 1
            miss = 0
        else:
            miss += 1
            if miss > 2 and abs(offset) > 30: #polvi ei voi olla minikapea sentään
                break
    return middledot, femurdot, tibiadot

# alustetaan kuva: pikselikoko, arvoavaruus, pienennetään
def init_xray_image(dicom_file):
    img, kerroin = scale(dicom_file, 0.15) #0.143
    img = img - img.min()
    img = img/img.max()*100 #skaalataan 100:n -> cannyssa kivempia lukuja
    img = rescale(img[:,100:int(img.shape[1]*0.5)-100], 0.25)
    img = gaussian_filter(img, 2)
    return img    
    

# Etsii oikean jalan (=vasemmalla kuvassa) polvialueen DICOM-tiedostosta
# Palauttaa
#    rajatun nivelalueen: knee_area
#    reisiluun alustuksen: init_area
#    polvi rajatun nivelalueen koordinaatit ap. kuvassa: knee_coord
#    tibian alustuksen: tibia_area
def find_knee_area(dicom_file):
    # alustetaan kuva 
    img = init_xray_image(dicom_file)
    
    # gradienttiversioita kuvasta
    chigh = 7
    clow = 1
    gcan = canny(img, sigma=0, low_threshold=clow, high_threshold=chigh)
    vsob = sobel_v(img)
    hsob = sobel_h(img)
    dpos = gcan*vsob #positiivinen vaakagradientti (01)
    dpos[dpos < 0] = 0
    dneg = gcan*vsob #neg. vaakagradientti (10)
    dneg[dneg > 0] = 0    
    gdir = np.arctan2(hsob, vsob) #gradientin suunta
    
    # etsitään karkeat luurajat (ei nivelrajaa)
    femur, ilist, jlist = find_femur(img, dpos, dneg)  
    # etsitään nivelraosta reisiluun reuna, sääriin reuna ja niiden puoliväli
    # polven suuntaisen gradientin avulla  
    middledot, femurdot, tibiadot = find_joint_space(img, femur, hsob, gdir)
    
    # rajataan alkuperäisestä kuvasta polvialue nivelrajojen perusteella
    ileft = np.min(np.nonzero(middledot)[1])*4
    iright = np.max(np.nonzero(middledot)[1])*4
    imid = int(np.mean(np.nonzero(middledot)[0]))*4
    halfwidth = int(0.3*(iright-ileft)) #0.3?
    img, _ = scale(dicom_file, 0.15)
    img = img[:,100:int(img.shape[1]*0.5)-100]
    femur_big = rescale(femurdot, 4)
    tibia_big = rescale(tibiadot, 4)
    
    knee_coord = [imid, halfwidth, ileft, iright]
    
    knee_area = img[imid-halfwidth : imid+halfwidth , ileft : iright]
    init_area = femur_big[imid-halfwidth : imid+halfwidth , ileft : iright]
    tibia_area = tibia_big[imid-halfwidth : imid+halfwidth , ileft : iright]
    if knee_area.size == 0:
        knee_area = img
    
    return knee_area, init_area, knee_coord, tibia_area



plt.close("all")


param_dict =  {'clow': [0.1], # 0.1-0.3
              'chigh': [0.3, 0.5, 0.6],# 0.9 yläraja 0.4-0.9
              'perlow': [10,15],
              'smoothing': [2,3],
              'lambda1': [2],
              'lambda2': [1]}


def thicken_line(img, n_iter):
    # s = np.array([[0],[0],[1]])
    s = np.ones((3,3))
    img = binary_dilation(img, structure=s, iterations=n_iter)
    return img


def list_params():
    
    param_dict =  {'clow': [0.5, 0.1, 0.15], # 0.1-0.3
                  'chigh': [0.5, 0.6, 0.7, 0.8],# 0.9 yläraja 0.4-0.9
                  'perlow': [10,15,20,25],
                  'smoothing': [2,3,4],
                  'lambda1': [1,2,3,4],
                  'lambda2': [1,2]}
    return ParameterGrid(param_dict)


def list_canny_params():
     param_dict =  {'clow': [0.03,0.05,0.1,0.2], # 0.1-0.3
                  'chigh': [0.3,0.4,0.5,0.6],# 0.9 yläraja 0.4-0.9
                  # 'perlow': [25,30,35,40],
                  'smoothing': [1,2,3,4]}
     return ParameterGrid(param_dict)   





# TODO
# lineaarisovitin eri pikselimäärille (päiden pituuksille) ja valitaan se,
# josta lyhin matka (vähiten pikseleitä) yhdistää seuraavaan
def join_and_interpolate(coor1, coor2, img):
    if coor1.size == 0 or coor2.size == 0:
        return np.concatenate((coor1, coor2), axis=0)

    left = extend_edge_left(coor1, img.shape)
    right = extend_edge_right(coor1, img.shape)
    
    # vasen pää
    # tarkastetaan, meneekö ylhäältä, alhaalta tai vasemmalta yli kuvan
    if 0 in left or img.shape[0]-1 in left[:,0]:
        left_extension = np.empty( shape=(0, 2) ).astype(int)
        left_joined = np.empty( shape=(0, 2) ).astype(int)
    else:
        #haetaan jatkeita lähimmät pisteet
        head1, head2 = nearest_kdtree(left, coor2)
        rr, cc = draw.line(head1[0], head1[1], coor1[0,0], coor1[0,1])
        left_extension =  np.stack((rr,cc)).transpose()
        rr, cc = draw.line(head2[0], head2[1], head1[0], head1[1])
        left_joined =  np.stack((rr,cc)).transpose()
    # oikea pää
    # tarkastetaan, meneekö ylhäältä, alhaalta tai oikealta yli kuvan
    if 0 in right or img.shape[1]-1 in right[:,1] or img.shape[0]-1 in right[:,0]:
        right_extension = np.empty( shape=(0, 2) ).astype(int)
        right_joined = np.empty( shape=(0, 2) ).astype(int)
    else:
        head1, head2 = nearest_kdtree(right, coor2)
        rr, cc = draw.line(coor1[-1,0], coor1[-1,1], head1[0], head1[1])
        right_extension =  np.stack((rr,cc)).transpose()
        rr, cc = draw.line(head1[0], head1[1], head2[0], head2[1])
        right_joined =  np.stack((rr,cc)).transpose()


    coor = np.concatenate((left_joined, left_extension, 
                           coor1, 
                           right_extension, right_joined), axis=0)
    edges =  np.zeros(img.shape)
    edges[coor1[:,0],coor1[:,1]] = 1
    edges[coor2[:,0],coor2[:,1]] = 1
    
    edges2 =  np.zeros(img.shape)
    edges2[coor[:,0],coor[:,1]] = 1

    return coor



#hakee kahden pistejoukon toisiaan lähimmät pisteet
#käyttää kd-puuta nopeaan hakuun
def nearest_kdtree(points, all_edge_points):
    edgetree = cKDTree(all_edge_points)
    dist, indexes = edgetree.query(points)
    imin = np.argmin(dist)
    return points[imin], all_edge_points[indexes[imin]]

def extend_edge_right(coor, img_shape):
    # suorasovite reunan päähän
    line = np.polyfit(coor[-10:,1], coor[-10:,0], 1)
    # reunan päätepikselit
    headx = int(coor[-1,1])
    heady = int(coor[-1,0])

    # piirretään 20 pikseliä
    # pythagoraan lauseella vaaka-kateetin pituus
    xl = int( math.sqrt((20**2)/(line[0]**2 + 1)) + headx )
    xl = min(xl, img_shape[1]-1) #poistetaan mahdolliset kuvan ulkopuoliset
    yl = int( line[0] * xl + line[1] )
    #poistetaan mahdolliset kuvan ulkopuoliset 
    rr, cc = draw.line(heady, headx, yl, xl)
    i = np.argwhere(np.logical_and(rr < img_shape[0], rr >= 0))
    rr = rr[i].transpose()[0]
    cc = cc[i].transpose()[0]
    extension_pixels = np.stack((rr,cc))
    
    extension_pixels = np.transpose(extension_pixels)
    return extension_pixels  


def extend_edge_left(coor, img_shape):
    # suorasovite reunan päähän
    line = np.polyfit(coor[:10,1], coor[:10,0], 1)
    # reunan päätepikselit
    headx = int(coor[0,1])
    heady = int(coor[0,0])

    # piirretään 20 pikseliä
    # pythagoraan lauseella vaaka-kateetin pituus
    x = int( headx - math.sqrt((20**2)/(line[0]**2 + 1)) )
    x = max(x, 0) #poistetaan mahdolliset kuvan ulkopuoliset
    y = int( line[0] * x + line[1] )
    #poistetaan mahdolliset kuvan ulkopuoliset 
    rr, cc = draw.line(heady, headx, y, x)
    i = np.argwhere(np.logical_and(rr < img_shape[0], rr >= 0))
    rr = rr[i].transpose()[0]
    cc = cc[i].transpose()[0]
    extension_pixels = np.stack((rr,cc))
    
    extension_pixels = np.transpose(extension_pixels)
    return extension_pixels   



def head(x):
    if x[4] == 1 and x.sum() == 2:
        return 1
    elif x[4] == 1 and x.sum() == 3:
        if (x[0] + x[1] == 2 or
            x[1] + x[2] == 2 or
            x[6] + x[7] == 2 or
            x[7] + x[8] == 2 or
            x[0] + x[3] == 2 or
            x[3] + x[6] == 2 or
            x[2] + x[5] == 2 or
            x[5] + x[8] == 2):
            return 1
    else:
        return 0
    return 0

#ottaa binäärikuvan ja hakee käyrän/reunan koordinaatit järjestyksessä vas->oik
def order_edge_coor(subedge):
    # etsitään ja järjestetään reunan päätepisteet
    heads = ndimage.filters.generic_filter(subedge, head, size=3)
    heads = np.argwhere(heads)
    heads = heads[heads[:,1].argsort()]
    # Jos ei löydy päätä -> palautetaan tyhjä taulukko
    if heads.size == 0:
        coordinates = np.array([])
        return coordinates

    # järjestetään reunapisteet
    r,c=np.where(subedge==1)
    z=list(zip(r,c))
    coorlist=[]
    coorlist.append(heads[0])
    for i in range(len(z)): #poimitaan aina edellistä lähin piste seuraavaksi
        pdist = distance.cdist(z,[coorlist[i]],'euclidean') #dist_all
        coorlist.append(z.pop(np.argmin(pdist))) #dist
    coorlist.pop(0)

    coordinates = np.asarray(coorlist)
    return coordinates

#sama kuin order_edge_coor, mutta palauttaa oikean pään
def order_left_edge_coor(subedge):
    # etsitään ja järjestetään alareunan päätepisteet
    if np.sum(subedge) < 2:
        heads = np.argwhere(subedge)
    else:
        heads = ndimage.filters.generic_filter(subedge, head, size=3)
        heads = np.argwhere(heads)
        heads = heads[heads[:,1].argsort()]

    # järjestetään reunapisteet
    r,c=np.where(subedge==1)
    z=list(zip(r,c))
    coorlist=[]
    coorlist.append(heads[0])
    for i in range(len(z)): #poimitaan aina edellistä lähin piste seuraavaksi
        pdist = distance.cdist(z,[coorlist[i]],'euclidean') #dist_all
        coorlist.append(z.pop(np.argmin(pdist))) #dist
    coorlist.pop(0)

    coordinates = np.asarray(coorlist)
    return coordinates, coordinates[-1]



# sama kuin order_edge_coor, mutta aloittaa edellist lähimmästä päästä sekä
# palauttaa loppupään
def order_relative_edge_coor(subedge, prev_end):
    if np.sum(subedge) < 2:
        heads = np.argwhere(subedge)
    else:
        heads = ndimage.filters.generic_filter(subedge, head, size=3)
        heads = np.argwhere(heads)
        
        pdist0 = distance.euclidean(prev_end, heads[0])
        pdist1 = distance.euclidean(prev_end, heads[1])
        if pdist0 > pdist1:
            heads = heads[[1,0],:]
    # järjestetään reunapisteet
    r,c=np.where(subedge==1)
    z=list(zip(r,c))
    coorlist=[]
    coorlist.append(heads[0])
    for i in range(len(z)): #poimitaan aina edellistä lähin piste seuraavaksi
        pdist = distance.cdist(z,[coorlist[i]],'euclidean') #dist_all
        coorlist.append(z.pop(np.argmin(pdist))) #dist
    coorlist.pop(0)

    coordinates = np.asarray(coorlist)
    return coordinates, coordinates[-1]


def order_all_edges(edge_image):
    s = [[1,1,1],
          [1,1,1],
          [1,1,1]]    
    labeled_edge_image, nedges = label(edge_image, structure=s)
    coor_all = np.empty( shape=(0, 2) )
    if nedges > 1:
        #järjestetään rajat eka toistensa suhteen
        lmean = []
        for edge in range(1, nedges+1):
            lmean.append(np.mean(np.nonzero(labeled_edge_image == edge)[1]))
        label_numbers = np.array(list(range(1, nedges+1)))
        label_numbers = label_numbers[np.argsort(lmean)]
        edge_no = label_numbers[0]
        segment = np.zeros(edge_image.shape)
        segment[labeled_edge_image == edge_no] = 1 
        coor, end = order_left_edge_coor(segment)
        coor_all = np.concatenate((coor_all, coor), axis=0)
        
        for edge_no in label_numbers[1:]: #range(1, nlines+1):
            segment = np.zeros(edge_image.shape)
            segment[labeled_edge_image == edge_no] = 1        
            coor, end = order_relative_edge_coor(segment, end)
            coor_all = np.concatenate((coor_all, coor), axis=0)
    else:
        coor_all = order_edge_coor(edge_image)    
    return coor_all


# Poistaa/tasaa ääreisarvot.
# Tasaa nivelraon kirkkauden sovittamalla polynomin ja vähentämällä sen arvot.
# Nivelraon kirkkaus vaihtelee pehmytkudoksen vaikutuksesta
# lähinnä horisontaalisuunnassa. Pehmytkudoksen vaikutus arvioidaan
# reisiluun alta nivelraosta (polynomisovitteella).
def equalize_joint_space(knee_area, init_area):
    knee_area_rsc = knee_area
    #nivelraon muotoa noudatteleva alustus vakaammaksi polynomiksi
    y = knee_area_rsc[np.nonzero(init_area*knee_area_rsc)]
    x = np.nonzero(init_area*knee_area_rsc)[1]
    fitted_poly = polyfit(x,y,4)
    y_poly = polyval(list(range(knee_area_rsc.shape[1])), fitted_poly)
    # vähennetään polynonilla arvioitu pehmytkudoksen vaikutus
    knee_area_eq = knee_area_rsc - y_poly
    # arvot välille 0-1
    knee_area_eq -= np.min(knee_area_eq)
    knee_area_eq = knee_area_eq/np.max(knee_area_eq)
      
    return knee_area_eq

def pikapiirto(img):
    plt.figure()
    plt.imshow(img, cmap=plt.cm.gray)
    plt.show()

def tibia_canny_line(img, cimg, femur_line):
    cimg_inc = cimg * sobel_h(img)    
    cimg_inc[cimg_inc < 0] = 0
    cimg_inc[cimg_inc > 0] = 1
    femur_line_coor = np.nonzero(femur_line)
    #mennään nyt forilla toistaiseksi
    #poistetaan femurin reunan yläpuoleiset pisteet
    j=0
    for i in femur_line_coor[1]:
        cimg_inc[ :femur_line_coor[0][j]+1 , i ] = 0
        j += 1
    skel = skeletonize(cimg_inc).astype(int)
    branch = find_branches(skel)
    skel[branch > 3] = 0
    top_skel = purge_low_branches(skel)
    return top_skel
    

def compare_points(points, amount=1):
    #normalisoidaan
    points_norm = []
    for p in points:
        points_norm.append(p/(np.max(p)+0.00001))
    points_norm = np.asarray(points_norm)
    sumpoints = np.sum(points_norm, axis=0)
    #suurimpien arvojen indeksit
    ind = np.argpartition(sumpoints, -amount)[-amount:]
    return ind



# alustetaan nivelrakoon arvauksia reisiluun reunan avulla
def initialize_joint_space(img, femur_edge):
    img = img - img.min()
    img = img/img.max()*100     
    chigh = 7
    clow = 1
    cs = 3
    # haetaan cannyn kanssa yhteinen viiva
    cimg = canny(img, sigma = cs, low_threshold=clow, high_threshold=chigh)
    init_area = femur_edge * cimg
    init_area[init_area != 0 ] = 1
    # karsitaan pienet rajaviivat/alueet todnäk virheinä pois
    s = [[1,1,1],
         [1,1,1],
         [1,1,1]]
    labeled_edge, labels =label(init_area, structure=s)      
    for i in range(labels+1):
        if np.sum(labeled_edge==i) < 20: #arvio sopivasta koosta
            init_area[labeled_edge==i] = 0
    # poimitaan koko rajaviiva
    labeled_canny, labels = label(cimg, structure=s)     
    for i in range(labels+1):
        if np.sum(init_area * labeled_canny==i) == 0:
            labeled_canny[labeled_canny==i] = 0
    init_area = labeled_canny
    
    #paksunnetaan alaspäin binäärilaajennuksella
    s = np.array([[0],[0],[1]])
    # s = np.ones((3,3))
    # init_area = binary_dilation(init_area, structure=s, iterations=4)
    # plt.figure("1")
    # plt.imshow(init_area)
    
    return init_area

# 
def initialize_tibia_line(img, femur_edge, params):
    chigh = params["chigh"]
    clow = params["clow"]
    cs = params["smoothing"]
    # haetaan cannyn kanssa yhteinen viiva
    cimg = canny(img, sigma = cs, low_threshold=clow, high_threshold=chigh)
    init_area = femur_edge * cimg
    init_area[init_area != 0 ] = 1
    # karsitaan pienet rajaviivat/alueet todnäk virheinä pois
    s = [[1,1,1],
         [1,1,1],
         [1,1,1]]
    labeled_edge, labels =label(init_area, structure=s)      
    for i in range(labels+1):
        if np.sum(labeled_edge==i) < 20: #arvio sopivasta koosta
            init_area[labeled_edge==i] = 0
    # poimitaan koko rajaviiva
    labeled_canny, labels =label(cimg, structure=s)     
    for i in range(labels+1):
        if np.sum(init_area * labeled_canny==i) == 0:
            labeled_canny[labeled_canny==i] = 0
    init_area = labeled_canny

    return init_area, cimg


def initialize_tibia_line2(img, init_area, params):
    chigh = params["chigh"]
    clow = params["clow"]
    cs = params["smoothing"]
    # haetaan cannyn kanssa yhteinen viiva
    cimg = canny(img, sigma = cs, low_threshold=clow, high_threshold=chigh)
    # karsitaan pienet rajaviivat/alueet todnäk virheinä pois
    s = [[1,1,1],
         [1,1,1],
         [1,1,1]]
    labeled_edge, labels =label(init_area, structure=s)      
    for i in range(labels+1):
        if np.sum(labeled_edge==i) < 5: #arvio sopivasta koosta
            init_area[labeled_edge==i] = 0
    # poimitaan koko rajaviiva
    labeled_canny, labels =label(cimg, structure=s)     
    for i in range(labels+1):
        if np.sum(init_area * labeled_canny==i) == 0:
            labeled_canny[labeled_canny==i] = 0
    return labeled_canny #init_area

# init_area = binäärikuva (karsittu canny)
def initialize_tibia_line3(img, init_area, params):
    chigh = params["chigh"]
    clow = params["clow"]
    cs = params["smoothing"]
    # haetaan cannyn kanssa yhteinen viiva
    cimg = canny(img, sigma = cs, low_threshold=clow, high_threshold=chigh)
    # karsitaan pienet rajaviivat/alueet todnäk virheinä pois
    s = [[1,1,1],
         [1,1,1],
         [1,1,1]]   
    # karsitaan rajaviivat, jotka eivät ole päällekkäisiä alustuksen kanssa
    labeled_canny, labels = label(cimg, structure=s)     
    for i in range(labels+1):
        if np.sum(init_area * labeled_canny==i) == 0:
            labeled_canny[labeled_canny==i] = 0
    
    tibia_line = labeled_canny * init_area
    tibia_line[tibia_line != 0] = 1
    return tibia_line #init_area


#etsii lähenytvät pisteet, jotka ovat yhteydessä vastakkaiseen pisteeseen:
#ensin lähin, ja sitten niiden pisteiden väliset pisteet
#TODO mitä jos pisteet haarautuvat? Toimii sattumanvaraisesti?
def search_nearest(coor, coor2, hlabel, labeled_img):
    if np.argwhere(labeled_img == hlabel).shape[0] == 1:
        return np.zeros(labeled_img.shape), np.empty( shape=(0, 2) )
    edgeimg = np.zeros(labeled_img.shape)
    edgeimg[labeled_img == hlabel] = 1
    subedge = skeletonize(edgeimg)
    subedge[coor2[0],coor2[1]] = 1 #jos vaikka tippuu skeletonin ulkopuolelle
    # joskus skeletonize karsii liikaa, jolloin kylmästi hylätään tämä
    _, n = label(subedge)
    if n > 1:
        return np.zeros(labeled_img.shape), np.empty((0,2))
    # järjestetään reunapisteet ja poimitaan järjestyksessä
    ordered_coorlist, _ = order_relative_edge_coor(subedge, coor)
    dists = distance.cdist([coor],ordered_coorlist,'euclidean') #dist_all
    inearest = np.argmin(dists) #dist
    icoor2 = np.argwhere((ordered_coorlist == coor2).all(axis=1))[0][0]
    if inearest < icoor2:
        nearing_coor = ordered_coorlist[inearest:icoor2]
    else:
        nearing_coor = ordered_coorlist[icoor2:inearest]
    img = np.zeros(labeled_img.shape) #tuloskuvamatriisi
    img[nearing_coor[:,0], nearing_coor[:,1]] = 1
    return img, nearing_coor
    

def create_circular_mask(h, w, pver, phor, r):
    y,x = np.ogrid[-pver:h-pver, -phor:w-phor]
    mask = x*x + y*y <= r*r
    return mask

def find_branches(img):
    s = [[1,1,1],
         [1,1,1],
         [1,1,1]]
    return convolve(img, s)*img


def join_via_canny(edges, img, img_shape, params):    
    chigh = params["chigh"]
    clow = params["clow"]
    cs = params["smoothing"]
    # haetaan cannyn kanssa yhteinen viiva
    cimg = canny(img, sigma = cs, low_threshold=clow, high_threshold=chigh)
    # karsitaan pienet rajaviivat/alueet todnäk virheinä pois
    s = [[1,1,1],
         [1,1,1],
         [1,1,1]]
    
    # järjestetään koordinaatit riveittäin omiin rivi- ja srk-vektoreihin
    # jokaisella rivillähän on maks. 1 piste
    coor = np.nonzero(edges)
    i = np.argsort(coor[1])
    coor_ver = coor[0][i]
    coor_hor = coor[1][i]
    # poimitaan pisteet, jotka eivät kosketa toisiaan, ts. L1-etäisyys > 1
    dist_hor = np.abs(coor_hor[1:] - coor_hor[:-1])
    dist_ver = np.abs(coor_ver[1:] - coor_ver[:-1])
    i = np.logical_or(dist_hor > 1, dist_ver > 1)
    ileft = np.concatenate((i,[False]))
    iright = np.concatenate(([False],i))
    #listataan pikseliparit (jotka eivät kosketa toisiaan)
    cleft_ver = coor_ver[ileft]
    cleft_hor = coor_hor[ileft]
    cright_ver = coor_ver[iright]
    cright_hor = coor_hor[iright]
    ihead = np.nonzero(ileft)[0] #päiden indeksit
    ihead = np.concatenate(([0],ihead))
    coorlist = np.empty( shape=(0, 2) ) #järjestetty koordinaattilista
    joined_img = np.copy(edges) #tuloskuvamatriisi
    for i in range(np.size(cleft_ver)): #jokainen pikselipari läpi
        cimg_masked = np.copy(cimg)
        # lasketaan vasemman ja oikean (pääte)pisteen välinen etäisyys
        pdist = distance.euclidean((cleft_ver[i], cleft_hor[i]), 
                                   (cright_ver[i], cright_hor[i])) + 0.01

        c = np.stack((coor_ver[ihead[i]:ihead[i+1]],coor_hor[ihead[i]:ihead[i+1]])).transpose()

        h, w = img.shape
        # luodaan lähestyvien reunapikseleiden tarkastelualue eli alue, jossa 
        # etäisyys päätepisteestä on lähempi kuin vastapuolen päätepisteen
        mask_left = create_circular_mask(h, w, cleft_ver[i], cleft_hor[i], 
                                         pdist)
        mask_right = create_circular_mask(h, w, cright_ver[i], cright_hor[i], 
                                          pdist)
        mask = mask_left * mask_right
        cimg_masked[~mask] = 0 #tarkastelualue
        labeled_mcimg, labels = label(cimg_masked, structure=s) 
        # päiden labelit
        lheadlabel = labeled_mcimg[cleft_ver[i], cleft_hor[i]]
        rheadlabel = labeled_mcimg[cright_ver[i], cright_hor[i]]
        
        lcoor = [cleft_ver[i], cleft_hor[i]]
        rcoor = [cright_ver[i], cright_hor[i]]
        redge_img, redge_coor = search_nearest(lcoor, rcoor, rheadlabel, labeled_mcimg)
        ledge_img, ledge_coor = search_nearest(rcoor, lcoor, lheadlabel, labeled_mcimg)
        joined_img += redge_img*2
        joined_img += ledge_img*3

        coorlist = np.concatenate((coorlist, c, ledge_coor, redge_coor), axis=0)

    joined_img[joined_img > 1] = 1
    img = joined_img
 
    
    return joined_img , coorlist 
        

# TODO vaihtoehto: raon intensiteetti 0:aan (alustuskäyrän avulla) +     
#      reisiluun intensiteetti a:han (esim. 1:een; alustuskäyrän avulla)
#      tai normaali histogrammin 0-1-skaalaus jokaiseen pikkukuvaan
#TODO poista haarat viimeisistä segmenteistä/reunoista
#TODO poista kirkkaimmat (ja tummimmmat) pikselit

#poistetaan päällekäisistä eri haaroihin kuuluvista pikseleistä alemmat
def purge_low_branches(skeleton_image):
    s = [[1,1,1],
         [1,1,1],
         [1,1,1]]
    labeled_skeleton, labels =label(skeleton_image, structure=s)      
    upmost_pixels = np.argmax(skeleton_image, axis=0)
    for i in range(labeled_skeleton.shape[1]): #sarakkeittain
        if len(np.unique(labeled_skeleton[:,i])) > 2: #jos väh. 2 eri haaraa
            upmost_value = labeled_skeleton[upmost_pixels[i],i]
            labeled_skeleton[labeled_skeleton[:,i] != upmost_value, i] = 0
    labeled_skeleton[labeled_skeleton > 0] = 1
    return labeled_skeleton

# poistetaan päällekäisistä eri haaroihin kuuluvista pikseleistä alemmat
# voisi kyllä käyttää yllä olevaa ylösalaiseen kuvaan vain
def purge_high_branches(skeleton_image):
    s = [[1,1,1],
         [1,1,1],
         [1,1,1]]
    labeled_skeleton, labels =label(skeleton_image, structure=s)      
    # ylin luurangon pikseli y-akselilla
    upmost_pixels = np.argmax(np.flipud(skeleton_image), axis=0)
    upmost_pixels = skeleton_image.shape[0] - 1 - upmost_pixels
    for i in range(labeled_skeleton.shape[1]): # sarakkeittain
        if len(np.unique(labeled_skeleton[:,i])) > 2: #jos väh. 2 eri haaraa
            upmost_value = labeled_skeleton[upmost_pixels[i],i]
            labeled_skeleton[labeled_skeleton[:,i] != upmost_value, i] = 0
    labeled_skeleton[labeled_skeleton > 0] = 1
    return labeled_skeleton

#poistetaan päällekäisistä pikseleistä alemmat
def purge_low_part(edge_image):
    upper_edges = np.copy(edge_image)
    upmost_pixels = np.argmax(upper_edges, axis=0)
    # käydään läpi sarakkeittain
    for i in range(upper_edges.shape[1]):
        # periaatteessa hajoo, jos pikseli alareunassa
        upper_edges[upmost_pixels[i]+1:,i] = 0 
    return upper_edges





def find_eminentia_edge(path):
    # path = "oai/baseline/blpolvi/all/"
    dirid = os.listdir(path)
    fpath = [os.path.join(path, imgid) for imgid in dirid]
    
    knee_area_list = []
    splineimg_list = [] 
    joined_tibia_line_list = []
    femur_edge_list = []
    spline_list = []
    
    params = {'smoothing': 2, 'perlow': 40, 'clow': 0.05, 'chigh': 0.15}
    i=0
    for fid in fpath:
        imgid = dirid[i]
        i += 1
        alku = 4001
        loppu = 4000
        if i < 4244:
            continue
        print("i = {}/{}, id = {}".format(i, loppu, imgid))
        fname = os.path.join(fid, imgid)
        ds = dicom.dcmread(fname)
        # etsitään ja rajataan polvialue
        knee_area, init_area, c, tibia_area = find_knee_area(ds)
 
        #tallennetaan löytynyt polvialue debuggausta helpottamaan
        plt.imsave("snd_tuloksia2/knee_area/"+imgid+".png", knee_area, cmap=plt.cm.gray)
        # poistutaan, jos epäonnistui alustus epäonnistuu
        if init_area.size == 0:
            print("Ei löytynyt niveltä. Jatketaan seur. kuvaan")
            continue 
        if sum(sum(init_area)) < 20:
            print("Ei löytynyt niveltä/femurin nivelpintaa. Jatketaan seur. kuvaan")
            continue
        if knee_area.shape[1] < 50:
            print("Löytynyt alue liian pieni. Jatketaan seur. kuvaan")
            continue
        # kirkkauden korjaus
        knee_area = equalize_joint_space(knee_area, init_area) 
        # p2, p98 = np.percentile(knee_area, (0.5, 99.5))
       
        tibia_line, cimg = initialize_tibia_line(knee_area, tibia_area, params)# param_grid[top_ind])   
        # haetaan reisiluun reuna
        femur_line = initialize_joint_space(knee_area, init_area)
        femur_line -= np.min(femur_line)
        femur_line = femur_line/np.max(femur_line)
        femur_line[femur_line != 0] = 1
        skel = skeletonize(femur_line).astype(int)
        branch = find_branches(skel)
        skel[branch > 3] = 0
        bottom_skel = purge_high_branches(skel) #poistetaan haarat
        femur_edge = bottom_skel
        
        top_cimg = tibia_canny_line(knee_area, cimg, bottom_skel)
        tibia_line = initialize_tibia_line2(knee_area, top_cimg, params)
        # bottom_skel = thicken_line(bottom_skel, 1)
        
        img = knee_area

        
        tibia_line[tibia_line != 0] = 1
        if sum(sum(tibia_line)) < 10:
            print("Ei löytynyt (>9 px) tibiarajaa. Jatketaan seur. kuvaan")
            continue            
        skel = skeletonize(tibia_line).astype(int)
        branch = find_branches(skel)
        skel[branch > 3] = 0
        
        cimg_inc = cimg * sobel_h(img)    
        cimg_inc[cimg_inc < 0] = 0
        cimg_inc[cimg_inc > 0] = 1

        s = [[1,1,1],
             [1,1,1],
             [1,1,1]]
        #yhdistetään alueiden rajaviivat
        labeled_level, nregions = label(skel, structure=s)
        if nregions > 1:
            coor_all = np.empty( shape=(0, 2) )
            for reg in range(1, nregions+1):   #label_numbers:  
                #poimitaan tulosalueen osa-alue
                segment1 = np.zeros(skel.shape)
                segment1[labeled_level==reg] = 1
                segment2 = skel - segment1
                if np.sum(segment1) < 5 or np.sum(segment2) < 5:
                    continue
                coor1 = order_edge_coor(segment1)#[::4]
                #TODO order_edge_coor ei käy järkeen segemnt2:lle. Mitä tämä yrittää tehdä?
                #noh, pitäis voida korvata ihan vain listaamisfunktioilla
                #nonzero tai where
                coor2 = order_edge_coor(segment2)#[::4]
                # Jos ei löydy päätä -> ei oteta mukaan
                if coor1.size == 0 or coor2.size == 0:
                    continue
                coor = join_and_interpolate(coor1, coor2,img+segment1+segment2)      
                coor_all = np.concatenate((coor_all, coor), axis=0)

            #joskus voi tulla 1 iso ja pari ylimääräistä pikseliä
            for reg in range(1, nregions+1):   #label_numbers
                if np.sum(labeled_level == reg) > np.sum(skel) - 5:
                    segment1 = np.zeros(skel.shape)
                    segment1[labeled_level==reg] = 1
                    coor_all = order_edge_coor(segment1)
                    break
        else:
            coor_all = order_edge_coor(skel)#[::4]skel
        if coor_all.size == 0:
            print("Tibiarajan iterointi epäonnistui. Jatketaan seur. kuvaan")
            continue
        coor_all = coor_all.astype(int)
        isort = np.argsort(coor_all[:,1])
        coor_all = coor_all[isort, : ]
        
        joined_edges = np.zeros(skel.shape)
        joined_edges[coor_all[:,0],coor_all[:,1]] = 1
        top_joined_edges = purge_low_part(joined_edges)
        tibia_line = initialize_tibia_line3(knee_area, top_joined_edges, params)
        # laajennetaan löydettyjä reunaviivoja cannyn avulla, koska nyt
        # jokaisessa x-akselin koordinaatissa on korkeintaan 1 reunapiste, eli
        # pystysuuntaiset reunat jäävät huomiotta
        joined_tibia_line, clist = join_via_canny(tibia_line, knee_area, knee_area.shape, params)
        joined_tibia_line = skeletonize(joined_tibia_line).astype(int)
        #spliniä varten järjestetyt pistet
        coor_all = order_all_edges(joined_tibia_line)
        y = coor_all[:,0]
        x = coor_all[:,1]
        tck,u = interpolate.splprep([x, y],s=0.15*x.size) #0.15 oli sopivan oloinen
        xi, yi = interpolate.splev(np.linspace(0, 0.95, 1000), tck)
        
        # poistetaan mahdollinen reunan yli menevät osuus
        if max(xi) >= skel.shape[1]:
            overind = np.where(xi.astype(int) >= skel.shape[1])[0][0]
            xi = xi[:overind]
            yi = yi[:overind]
        if max(yi) >= skel.shape[0]:
            overind = np.where(yi.astype(int) >= skel.shape[0])[0][0]
            xi = xi[:overind]
            yi = yi[:overind] 
        #piirretään splini kuvamatriisiin
        splineimg = np.zeros(skel.shape)
        splineimg[yi.astype(int),xi.astype(int)] = 1
        


        init_area = binary_dilation(init_area, structure=s, iterations=4)
        
        drawimg = np.array([img, img, img])
        drawimg = np.transpose(drawimg, (1,2,0)) 
        drawimg = drawimg/np.max(drawimg)*2 #skaalataan arvot välille 0-2
        # drawimg[:,:,0] += b
        # drawimg[:,:,1] += joined_tibia_line#top_joined_edges #init_area/np.max(init_area)
        drawimg[:,:,1] += splineimg #joined_tibia_line
        drawimg[:,:,2] += femur_edge*1.5 #joined_tibia_line*2
        drawimg[:,:,1] += 0.5*femur_edge
        # drawimg[yi.astype(int), xi.astype(int), 0] = 1
        # drawimg[:,:,1] += init_area #cimg #splineimg #skel #bottom_skel #femur_line
        # drawimg[corners[:,:,0].astype(int),corners[:,:,1].astype(int),0] += 1 #tibia_line
        drawimg = drawimg/np.max(drawimg) #skaalataan arvot välille 0-1 uudestaan
        # drawimg2 = np.array([img, img, img])
        # drawimg2 = np.transpose(drawimg2, (1,2,0))
        # drawimg2 = 0.67*drawimg2
        # drawimg3 = np.concatenate((drawimg2, drawimg))
        # plt.imsave("snd_tuloksia/radiologikuva"+str(i)+".png", drawimg3)  
        plt.imsave("snd_tuloksia2/"+imgid+".png", drawimg)  
        
           
        print("piirretty kuva {}/{}".format(i, len(fpath)))
        
        knee_area_list.append(knee_area)
        splineimg_list.append(splineimg)
        joined_tibia_line_list.append(joined_tibia_line)
        femur_edge_list.append(femur_edge)
        spline_list.append(tck)

    return knee_area_list, spline_list, splineimg_list, joined_tibia_line_list, femur_edge_list




if __name__ == '__main__':
    img, splineimg, joined_tibia_line, femur_edge = find_eminentia_edge("oai/baseline/blpolvi/all/")
    
 

  