# Kasvain-strooma suhdeluku / Tumor-stroma ratio

Kasvain-strooma suhdeluku (TSR) on yksi niistä tekijöistä, joita patologi määrittää tutkiessaan histopatologista kasvaimesta otettua näytettä mikroskoopilla. Suuri strooman osuus kasvaimen alueella viittaa huonoon ennusteeseen. TSR arvioidaan kasvaimen alueelta silmämääräisesti kohdasta, jossa strooman osuus on suurin. Haasteena tässä on toistettavuus.

*Tumor-stroma ratio (TSR) is one of the factors pathologists estimates when analyzing a histopathogical sample from cancer tissue on a microscope. The greater the amount of stroma is in the tumor site, the worse is the prognosis of the patient. Visual TSR is scored from a spot, which seems to have the biggest amount of stroma. As the process is manual, the challenge is the reproducibility of this scoring method.*

### Silmämääräinen kasvain-strooma suhdeluvun arviointi / Visual estimation of tumor-stroma ratio
<img width="401" alt="tsr_example" src="https://user-images.githubusercontent.com/64031196/165466014-5ffd43e3-434b-413a-a074-f4cca997421a.png">

TSR arvioidaan kasvaimen alueelta kohdasta, jossa on eniten stroomaa. Näkymässä tulee olla vähintään neljällä laidalla kasvainta.

*TSR is estimated from tumor site from a spot, where the amount of stroma is the greatest. Tumor must be visible on at least four sides of the chosen view.*

Kuva: Van Pelt, et al. "Scoring the tumor-stroma ratio in colon cancer: procedure and recommendations." Virchows Archiv 473, no. 4 (2018): 405-412.

### Automaattinen kasvain-strooma suhdeluvun arviointi / Automated estimation of tumor-stroma ratio

<img width="601" alt="automated_TSR" src="https://user-images.githubusercontent.com/64031196/165464784-2a23dd50-f94a-471a-b37e-1f0308c3623b.png">

Kun TSR ennustetaan automaattisesti, koko näytteen alue pilkotaan pienempiin kuvatiiliin ja malli ennustaa kullekin kuvatiilelle luokan. Luokkia on kolme: strooma, kasvain, muu. Laskemalla strooman osuus kasvain- ja stroomatiilten kokonaismäärästä, saadaan TSR.

*When predicting TSR automatically, entire sample is tiled to smaller image tiles and the model predicts class for each tile. There are three classes: stroma, tumor and other. TSR is the percentage of stroma tiles from the total amount of stroma plus tumor tiles.*

# Konvoluutioneuroverkkomallit / Convolutional neural network models

Kansiossa **models** on saatavilla kolme erilaista paksusuolensyövästä otetuilla histopatologisilla kuvilla opetettua konvoluutioneuroverkkoluokitinta.

- **SETUP_1_vgg19_FINAL.pt**: 
    - esiopetus: ImageNet ja Kather et. al. (2018) julkaisema datasetti
    - lopullinen opetus: "Suolisyöpä Keski-Suomessa 2000-2015" -hankkeen kuvat
    
- **SETUP_2_vgg19_FINAL.pt**:
    - esiopetus ImageNet
    - lopullinen opetus: "Suolisyöpä Keski-Suomessa 2000-2015" -hankkeen kuvat
    
- **SETUP_3_googlenet_FINAL.pt**: 
    - esiopetus Kather et. al. (2018) julkaisema datasetti
    - lopullinen opetus: "Suolisyöpä Keski-Suomessa 2000-2015"-hankkeen kuvat
    
Luokat:

- **0**: muu
- **1**: strooma
- **2**: kasvain

---

*Three CNNs trained with histopathological images from colorectal cancer can be downloaded from the **models** -folder.*

- **SETUP_1_vgg19_FINAL.pt**: 
    - *pre-training: ImageNet and Kather et. al. (2018) colorectal cancer public dataset*
    - *final training with images from "Suolisyöpä Keski-Suomessa 2000-2015" project*
    
- **SETUP_2_vgg19_FINAL.pt**:
    - *pre-training: ImageNet*
    - *final training with images from "Suolisyöpä Keski-Suomessa 2000-2015" project*
    
- **SETUP_3_googlenet_FINAL.pt**: 
    - *pre-training: Kather et. al. (2018) colorectal cancer public dataset*
    - *final training with images from "Suolisyöpä Keski-Suomessa 2000-2015" project*
    
*Classes*:

- **0**: *other*
- **1**: *stroma*
- **2**: *tumor*


## Syötekuvat / Input images

- syötekoko kaikille malleille 224 x 224 px$^2$
- mallit on koulutettu Macenko-normalisoiduilla kuvilla

- SETUP_1 - ja SETUP_2 -mallit:
    - kuvien normalisointiin käytettävät keskiarvot = [0.485, 0.456, 0.406] ja keskihajonnat = [0.229, 0.224, 0.225]
    - käytä vgg19-arkkitehtuuria, joka on esiopetettu ImageNetillä

- SETUP_3 -malli: 
     - kuvien normalisointiin käytettävät keskiarvot = [0.6979, 0.4694, 0.6644] ja keskihajonnat = [0.1371, 0.1540, 0.1269]
     - käytä googlenet-arkkitehtuuria, ilman ImageNet-esiopetusta

---

- *input size of all models is 224 x 224 px$^2$*
- *models have been trained with Macenko-normalized images*

- *SETUP_1 - and SETUP_2 -models:*
    - *means and standard deviations for the normalization of the images are: means = [0.485, 0.456, 0.406] and stds = [0.229, 0.224, 0.225]*
    - *use vgg19-architecture pre-trained with ImageNet*
    
- *SETUP_3 -model:* 
    - *means and standard deviations for the normalization of the images are: means = [0.6979, 0.4694, 0.6644] and stds = [0.1371, 0.1540, 0.1269]*
    - *use googlenet-architecture **without ImageNet-pre-training***
