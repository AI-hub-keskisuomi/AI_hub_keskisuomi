import math
from random import gauss
import numpy as np

from skimage.transform import rescale
from skimage.feature import canny #reunantunnistus
from skimage.filters import sobel_v, sobel_h #reunantunnistus

from scipy.ndimage import gaussian_filter
from scipy import stats #tilasto- ja todnäk-funktioita 


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

# Etsii oikean jalan (=vasemmalla kuvassa) polvialueen DICOM-tiedostosta
# Palauttaa
#    rajatun nivelalueen: knee_area
#    reisiluun alustuksen: init_area
#    polvi rajatun nivelalueen koordinaatit ap. kuvassa: knee_coord
#    tibian alustuksen: tibia_area
def find_knee_area(img_orig):
    # alustetaan kuva 
    # gradienttiversioita kuvasta
    img=img_orig.copy()
    scaled_width=250
    scale_factor=img.shape[1]/scaled_width
    img=gaussian_filter(img,2)
    img=rescale(img, 1/scale_factor)
    clow = 2
    chigh = 6
    c_smoothing=1
    img-=img.min()
    img=100*img/img.max()
    cimg = canny(img, sigma=c_smoothing, low_threshold=clow, high_threshold=chigh)
    
    vsob = sobel_v(img)
    hsob = sobel_h(img)
    dpos = cimg*vsob #positiivinen vaakagradientti (01)
    dpos[dpos < 0] = 0
    dneg = cimg*vsob #neg. vaakagradientti (10)
    dneg[dneg > 0] = 0    
    gdir = np.arctan2(hsob, vsob) #gradientin suunta
    
    # etsitään karkeat luurajat (ei nivelrajaa)
    femur, _, _ = find_femur(img, dpos, dneg)  
    # etsitään nivelraosta reisiluun reuna, sääriin reuna ja niiden puoliväli
    # polven suuntaisen gradientin avulla  
    middledot, _, _ = find_joint_space(img, femur, hsob, gdir)
    
    # rajataan alkuperäisestä kuvasta polvialue nivelrajojen perusteella
    ileft = np.min(np.nonzero(middledot)[1])*scale_factor
    iright = np.max(np.nonzero(middledot)[1])*scale_factor
    imid = np.mean(np.nonzero(middledot)[0])*scale_factor
    halfwidth = 0.3*(iright-ileft)
    
    return [int(c) for c in (imid-halfwidth,imid+halfwidth,ileft,iright)]