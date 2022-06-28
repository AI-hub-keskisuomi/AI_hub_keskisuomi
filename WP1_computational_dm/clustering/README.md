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
5. Klusterointi toistettiin vaihtelemalla ryhmien lukumääräarvoa väliltä [2, 10]. Alkupisteiden valinnassa käytettiin K-means++-klusterointia ja kussakin klusteroinnissa käytettiin sataa replikaattia. 
6. Kahdeksan tunnettua klusterivalidointi-indeksiä valittiin validoimaan klusterien lukumäärätietoja.   

#### Tulokset

Klusterivalidointi-indekseistä PBM ja WB suosittelivat alkumittausdatalle kolmea eri ryhmää ja tämä valittiin ryhmien lopulliseksi lukumääräksi. 

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

Tarkastelut suoritettiin myös loppumittausdatalle, josta indeksit löysivät peräti seitsemän profiilia. Loppumittausaineiston tuloksissa oli selkeästi enemmän hajontaa profiilien välillä. Kaksi näistä profiileista oli pieniä, koska molemmat profiilit sisälsivät ainoastaan kolme potilasta. Kahdessa parhaimmassa profiilissa potilaat (yhteensä 19 kpl) kokivat terveytensä keskimääräistä paremmaksi ja profiilien potilaat saivat pudotettua painoa yli 2,5 %. Profiilissa 3 potilaat (23 kpl) tunsivat terveytensä keskimääräistä paremmaksi, mutta heillä oli tästä huolimatta havaittavissa suhteellista painon nousua koko 36 kk tarkastelujaksolta. Profiilissa 4 potilaat (17 kpl) puolestaan tunsivat terveytensä keskimääräistä huonommaksi, mutta onnistuivat pudottamaan painoaan noin 2 %. Profiilissa 6 potilaat (13 kpl) tunsivat terveytensä keskimääräistä huonommaksi ja heillä oli havaittavissa suhteellista painon nousua tarkastelujaksolta. Kaikki klusterointitulokset ovat saatavilla tiedostossa 'Klusterointituloksia.xlsx'.     
