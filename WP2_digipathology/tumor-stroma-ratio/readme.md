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

Kansiossa **models** on saatavilla kolme erilaista paksusuolensyövästä otetuilla histopatologisilla kuvilla koulutettua konvoluutioneuroverkkoluokitinta:

- **SETUP_1_vgg19_FINAL.pt**: esiopetettu ImageNetillä sekä Kather et. al. (2018) julkaisemalla datasetillä, lopullinen opetus tehty "Suolisyöpä Keski-Suomessa 2000-2015" -hankkeen kuvilla
- **SETUP_2_vgg19_FINAL.pt**: esiopetettu ImageNetillä, lopullinen opetus tehty "Suolisyöpä Keski-Suomessa 2000-2015" -hankkeen kuvilla
- **SETUP_3_googlenet_FINAL.pt**: esiopetettu Kather et. al. (2018) julkaisemalla datasetillä, lopullinen opetus tehty "Suolisyöpä Keski-Suomessa 2000-2015"-hankkeen kuvilla

Luokat:

- **0**: muu
- **1**: strooma
- **2**: kasvain
