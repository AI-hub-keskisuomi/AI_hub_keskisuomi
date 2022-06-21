## Psykologisten tekijöiden vaikuttavuuden arvoinointi Muutosmatka-interventiossa
### Case 1: Piirteiden tärkeysarvojen tarkastelu

Psykologiset tekijät voidaan jakaa neljään kategoriaan:
- Psykologinen joustavuus (AAQ-II)
- Ajattelutoiminnot (WBSI)
- Minäpystyvyys (GSE)
- Mieliala (DASS-21)

Potilaille on mahdollista laskea pyskologisten tekijöiden pisteytykset kyselylomakekyselyiden perusteella. Alhaiset arvot AAQ, WBSI ja DASS pisteytyksissä ja korkeat arvot GSE-pistetyksessä kuvaavat hyviä tuloksia. Osalla muuttujista on olemassa myös alakategorioita. 

Terveyteen liittyväksi riskitekijäksi valittiin kohdehenkilöiden painotiedot ja psykologisten tekijöitä verrattiin painotietoihin. Tavoitteena oli pyrkiä tunnistamaan, mitkä psykologiset tekijät vaikuttavat eniten painon muutoksiin. Tuloksia oli mahdollista tarkastella tilastollisesti korrelaatioanalyyseillä ja koneoppimislähtöisesti piirteiden tärkeysarvojen tarkasteluilla. Tietyillä luokittelu- ja regressiomalleilla on mahdollista laskea painoarvot piirteille, jotka maksimoivat ennustustarkkuuden. Tällaisia malleja ovat esimerkiksi satunnaismetsä (eng. random forest) ja "Extremely randomized trees". Nämä kaksi mallia valittiin mukaan tarkasteltaviksi.

Tärkeysarvojen laskennat sisälsivät seuraavat vaiheet:
1. Potilaiden rajaus niihin henkilöihin, jotka olivat mukana koko Muutosmatka-intervention ajan eli 36 kk. Potilasmäärä rajautui 78 potilaaseen.  
2. Suhteellisten painon muutosten laskenta intervention alusta intervention loppuun.
3. Psykologisten pisteytyksien muutoksien laskenta yksittäisille psykologisille muuttujille ja neljän pääkategorian summamuuttujille intervention alusta intervention loppuun.
4. Puuttuvien arvojen käsittely. Puuttuvat summamuuttujat imputoitiin mediaani-imputoinnilla ja puuttuvien yksittäisten muuttujien imputoinnissa käytettiin 10-lähimmän naapurin imputointia. 
5. Datan skaalaus nollakeskiarvoiseksi ja yksikköhajonnalle Z-score muunnoksella.
6. Verkkohakua (eng. Grid search) oli mahdollista käyttää mallien parametrien optimointeihin.
7. Koneoppimismalleihin liittyy mallien satunnaisuuteen perustuvaa laskennallista epävakautta. Tämän vuoksi suoritettiin yhteensä 100 uudelleen ajoa. Lisäksi malleihin liittyy riski ylioppia aineisto, jonka vuoksi tulokset voivat vaikuttaa hyviltä mikäli opetukseen ja testaukseen käytetään samaa dataa. Ristiinvalidointi on keino ehkäistä ylioppimista. Ideana on pilkkoa data opetus- ja testiaineistoksi määrätty määrä kertoja siten että kukin näyte on testiaineistossa tasan yhden kerran. Työssä käytettiin viisinkertaista ristiinvalidointia, jossa datan jaottelu suoritettiin yhteensä viisi kertaa kullakin ajolla.    
8. Tärkeysarvot keskiarvoistettiin yhdellä ajolla jakamalla viidellä.
9. Tärkeysarvot keskiarvoistettiin sadan uudelleen ajon jälkeen. Lisäksi arvoille laskettiin keskihajonta ja mediaani.    

Saadut tulokset perustuivat regressioennusteisiin siten, että painon suhteellista muutosta ennustettiin jatkuvana muuttujana. Lisäksi mallien suorituskykyä tarkasteltiin luokittelussa siten, että ennustettavana muuttujana oli luokkatieto painon suhteellisesta muutoksesta (0 = painon suhteellinen muutos on alle 2,5 % ja 1 = painon suhteellinen muutos on yli 2,5 %). 

#### Tulokset

Tulosten todenperäisyyttä oli mahdollista arvioida käyttämällä kahta eri mallia ja vertailla mallien tuloksia. Toinen lähestymistapa tulosten arviointeihin oli laskea mallien ennustustarkkuudet sekä luokittelu- että regressiotehtävässä. 

Summamuuttujien tärkeysarvojen tulokset satunnaismetsäluokittelumallilla:
---  |  | Mean | (Std) | Median | 
--- | --- | --- | --- |--- |
1 | WBSI | 0,342  | (0,008)  | 0,343  | 
2 | AAQ | 0,247 | (0,006) | 0,248 | 
3 | DASS | 0,230  | (0,006)  | 0,230 | 
4 | GSE | 0,181  | (0,005)  | 0,181 | 

Summamuuttujien tärkeysarvojen tulokset Extra Tree -luokittelumallilla:
---  |  | Mean | (Std) | Median | 
--- | --- | --- | --- |--- |
1 | WBSI | 0,297 | (0,005) | 0,298 | 
2 | AAQ | 0,254 | (0,005)  | 0,254 | 
3 | DASS | 0,244  | (0,005) | 0,244 | 
4 | GSE | 0,205 | (0,004) | 0,204 | 

Summamuuttujien tärkeysarvojen tulokset satunnaismetsäregressiomallilla:
---  |  | Mean | (Std) | Median | 
--- | --- | --- | --- |--- |
1 | WBSI | | |  | 
2 | AAQ | | | | 
3 | DASS | | | | 
4 | GSE | | | | 

Summamuuttujien tärkeysarvojen tulokset Extra Tree -regressiomallilla:
---  |  | Mean | (Std) | Median | 
--- | --- | --- | --- |--- |
1 | WBSI | | | | 
2 | AAQ | | | | 
3 | DASS | | | | 
4 | GSE | | | | 


