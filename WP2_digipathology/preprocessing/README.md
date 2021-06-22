### Histopatologisten kuvien esikäsittely / _Preprocessing of histopatological Whole Slide Images_

Tekoälymallien opettamiseen tarvitaan paljon dataa. Konenäön tapauksessa digipatologiassa käytetään digitoituja, histopatologisista näytteistä skannattuja kuvia (Whole Slide Image, WSI). WSI-kuvat eivät ole sellaisenaan kuitenkaan ole valmiita tekoälymallin opettamiskäyttöön, sillä ne koostuvat biljoonista pikseleistä ja voivat sisältää useita leikkeitä sekä paljon taustaa. Esikäsittelyvaiheessa kuvat pilkotaan pieniin noin 32-256 x 32-256 pikselin kokoisiin palasiin ja vain ne palaset, joissa kudosta on ainakin 80-90 %, valitaan eteenpäin. 

Koska kerätyt kuvat voivat olla peräisin eri laboratorioista, käytössä on voinut olla erilaisia skannereita ja mahdollisesti toisistaan poikkeavia leikkeiden värjäysproseduureja, kuvien palastamisen jälkeen tarvitaan usein värien normalisointia. Tähän on olemassa useita eri menetelmiä, joista osa on laskennallisesti vaativampaa, tekoälyyn pohjautuvaa normalisointia, osa perustuu normalisoitavan kuvan skaalaamiseen käyttäen mallikuvan väriavaruutta pohjana laskennassa.

_Preprocessing histopatological images plays an important role in developing machine learning models, as it can affect the performance and reliability of trained systems, as well as qualitative analysis and validation. Tissue samples are processed to small biopsy glasses, which are then scanned digitally. This procedure creates huge amounts of data in the form of Whole Slide Images (WSI), one WSI could have as many as billions of pixels. From these WSIs pathologists make their own analysis of the cancer type and state. The biopsy process is mainly manual work, therefore biological heterogeneity is not the only thing that makes each WSI differ from each other. Also the stain colour, sample size, shape and the number of tissue specimens per slide vary a lot._

_In order to give the set of WSIs to the machine learning algorithm, the slides have to be preprocessed. WSIs are first tiled to small 32-256 x 32-256 pixel tiles and tiles which have 80-90 % tissue are chosen. After tiling step, the color is normalized, this step is important especially if the samples are from different laboratories which might have different staining methods or scanners._

![wsi_preprocess](https://user-images.githubusercontent.com/64031196/112951709-4cf72100-9144-11eb-8bae-d53f93310707.png)
1. Yhdessä histopatologisesta näytteestä skannatussa WSI-kuvassa voi olla yksi tai useampi leike sekä valkoista taustaa. Kuvan WSI on n. 60 000 x 60 000 pikseliä. / _One WSI can include one or more specimen and background. This WSI is about 60 000 x 60 000 pixels._
2. WSI pilkotaan noin 32-256 x 32-256 kokoisiin palasiin. / _WSI is tiled to small tiles, about 32-256 x 32-256 pixels._
3. Jatkoon valitaan palaset, joissa on kudosta tai annotoituja aluetta 80-90 % / _Tiles which have 80-90 % tissue are chose forward._
4. Kun kaikki kuvat on muutettu halutun kokoisiksi tiiliksi, värit normalisoidaan. / _After tiling, colors are normalized._
5. Kuvapalaset ovat valmiita käytettäväksi tekoälymallin opettamiseen. / _Tiles are now ready for the machine or deep learning algorithm._
