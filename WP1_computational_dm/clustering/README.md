## Psykologisten tekijöiden vaikuttavuuden arviointi Muutosmatka-interventiossa
### Case 2: Klusterointi
Psykologiset tekijät voidaan jakaa neljään kategoriaan:
- Psykologinen joustavuus (AAQ-II)
- Ajattelutoiminnot (WBSI)
- Minäpystyvyys (GSE)
- Mieliala (DASS-21)

Potilaille oli mahdollista laskea psykologisten tekijöiden pisteytykset kyselylomakekyselyiden perusteella. Alhaiset arvot AAQ, WBSI ja DASS pisteytyksissä ja korkeat arvot GSE-pistetyksessä kuvaavat hyviä tuloksia. Klusteroinnilla pyrittiin ryhmittelemään psykologisia muuttujia samankaltaisiin ryhmiin alkumittauksessa ja loppumittauksessa. Klusterien lukumäärät valittiin klusterivalidointi-indekseillä. Saaduille klusteriprofiileille laskettiin metadataa, kuten keskiarvo- ja mediaanipainot, suhteelliset painon muutokset loppumittauksessa ja koetun terveyden kolmiportaiset pisteytykset kussakin profiilissa. Itsearviointikyselyssä ykköset kuvasivat hyvää koettua terveyttä, kakkoset kuvasivat kohtalaista terveyttä ja kolmoset kuvasivat huonoa terveyttä. Alkumittausdata laskettiin 177 henkilöltä ja loppumittausdata 78 henkilöltä.  

Klusterointi sisälsi seuraavat vaiheet:
1. Puuttuvia arvoja oli vain vähän, joten datan täydentämisessä käytettiin mediaani-imputointia.
2. Viisiportainen koettu terveys skaalattiin kolmiportaiseksi.
3. Data normalisoitiin Min-max-skaalauksella välille [-1, 1].
4. Klusteroinnissa käytettiin K-spatialmedians-klusterointia.
5. Klusterointia toistettiin vaihtelemalla ryhmien lukumääräarvoja väliltä [2, 10]. Alkupisteiden valinnassa käytettiin K-means++-algoritmia ja kussakin klusteroinnissa käytettiin yhteensä sataa replikaattia. 
6. Kahdeksan tunnettua klusterivalidointi-indeksiä valittiin validoimaan klusterien lukumäärätietoja.   

#### Tulokset

##### Alkumittausdata 

Klusterivalidointi-indekseistä Pakisha-Bandaya-Maulik ja WB-indeksi suosittelivat alkumittausdatalle kolmea eri ryhmää ja tämä valittiin ryhmien lopulliseksi lukumääräksi. 

Psykologisten muuttujien mediaaniarvot kussakin profiilissa olivat seuraavia:
---  | Profiili 1 (N=56)  | Profiili 2 (N=80) | Profiili 3 (N=41) | Kaikki (N=177) 
---  | ---  | ---  | ---  | ---  | 
AAQ  | 8,5  | 15,0  | 28,0  | 15,0  | 
DASS  | 4,0  | 9,0  | 19,0  | 8,0  |
GSE  | 35,0  | 28,0  | 28,0  | 30,0  | 
WBSI  | 27,5  | 40,5  | 56,0  | 40,0  | 

Potilaat jakautuivat psykologisten muuttujien perusteella selkeästi kolmeen eritasoiseen ryhmään. 

Profiilien 1 ja 2 potilaat olivat hieman hoikempia verrattuna profiilin 3 potilaisiin ja he kokivat terveytensä paremmaksi kuin keskimäärin profiilissa 3. Tämä ilmenee seuraavasta taulukosta:
---  | Profiili 1 (N=56)  | Profiili 2 (N=80) | Profiili 3 (N=41) | Kaikki (N=177) 
---  | ---  | ---  | ---  | ---  | 
terv_tila  |  |   |   |   | 
1  | 34  | 42  | 7  | 83  | 
2  | 14  | 29  | 19  | 62  | 
3  | 8  | 9  | 15  | 32  | 
Keskiarvo  | 1,5357  | 1,5875 | 2,1951  | 1,7119  | 

##### Loppumittausdata

Tarkastelut suoritettiin myös loppumittausdatalle, josta indeksit (yhteensä 7 indeksiä) löysivät peräti seitsemän profiilia (kuvat indeksien tuloksista ovat Jpeg-kansiossa). Loppumittausaineiston tuloksissa oli selkeästi enemmän hajontaa profiilien välillä. Kaksi näistä profiileista oli pieniä, koska molemmat profiilit sisälsivät ainoastaan kolme potilasta. Kahdessa parhaimmassa profiilissa potilaat (yhteensä 19 kpl) kokivat terveytensä keskimääräistä paremmaksi ja profiilien potilaat saivat pudotettua painoa yli 2,5 %. Profiilissa 3 potilaat (23 kpl) tunsivat terveytensä keskimääräistä paremmaksi, mutta heillä oli tästä huolimatta havaittavissa suhteellista painon nousua koko 36 kk tarkastelujaksolta. Profiilissa 4 potilaat (17 kpl) puolestaan tunsivat terveytensä keskimääräistä huonommaksi, mutta onnistuivat pudottamaan painoaan noin 2 %. Profiilissa 6 potilaat (13 kpl) tunsivat terveytensä keskimääräistä huonommaksi ja heillä oli havaittavissa suhteellista painon nousua tarkastelujaksolta. 

Kaikki klusterointitulokset ovat saatavilla tiedostossa 'Klusterointituloksia.xlsx'.     

## Effectivity analysis of psychological variables in a lifestyle intervention
### Case 2: Clustering

Psychological variables can be divided into the four categories:
- Psychological flexibility (AAQ-II)
- Thought suppression (WBSI)
- Self-efficacy (GSE)
- Psychological distress (DASS-21)

It was possible to compute points of psychological behavior for each category by using questionnaires. Lower values in AAQ, WBSI, and DASS and higher values in GSE indicated good results, respectively. The aim of the clustering is to divide the data into distinct groups where data samples are similar within a group and dissimilar to other samples in a different group. The clustering was performed based on the data set measured from the beginning of intervention and the data set measured from the end of the intervention. The number of the clusters was decided by using the internal cluster validation indices. The metadata was computed for the obtained cluster profiles, e.g., mean and median weights, relative weight changes, and self-rated healthiness were measured for the each profile. Three options was given to the healthiness (good healty = 1, stable healthy = 2, and bad healthy = 3). There were 178 patients who started the intervention and 78 patients who completed the whole 36-montly test period.           

The clustering included the following steps:
1. There were only few missing values which were imputed by the median imputation.
2. The self-rated healty originally consisted of five ratings but the measure was scaled to three ratings.
3. Data was Min-max to the range of [-1, 1].
4. Data was clustered by using K-spatialmedians clustering.
5. The clustering was performed for different numbers of clusters from the range of [2, 10]. The initial points were selected by using K-means++ algorithm and the clustering was replicated by using 100 replicates. 
6. The clustering results were validated with eight well-known cluster validation indices.

#### Results

##### Results using the data at the beginning of the intervention  

Klusterivalidointi-indekseistä Pakisha-Bandaya-Maulik ja WB-indeksi suosittelivat alkumittausdatalle kolmea eri ryhmää ja tämä valittiin ryhmien lopulliseksi lukumääräksi. 




