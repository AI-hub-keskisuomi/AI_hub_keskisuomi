## Psykologisten tekijöiden vaikuttavuuden arviointi Muutosmatka-interventiossa

#### Piirteiden tärkeysarvojen tarkastelu

Psykologiset tekijät voidaan jakaa neljään kategoriaan:
- Psykologinen joustavuus (AAQ-II)
- Ajattelutoiminnot (WBSI)
- Minäpystyvyys (GSE)
- Mieliala (DASS-21)

Muutosmatka-interventioon osallistuneille oli mahdollista laskea psykologisten tekijöiden pisteytykset lomakekyselyiden perusteella. Alhaiset arvot AAQ, WBSI ja DASS pisteytyksissä ja korkeat arvot GSE-pistetyksessä kuvaavat hyviä tuloksia. Osalla muuttujista on olemassa myös alakategorioita. Terveyteen liittyväksi riskitekijäksi valittiin kohdehenkilöiden painotiedot ja psykologisten tekijöitä verrattiin painotietoihin. Tavoitteena oli pyrkiä tunnistamaan ne psykologiset tekijät, jotka vaikuttavat eniten painon muutoksiin. 

#### Koneoppimismallit 

Tietyillä luokittelu- ja regressiomalleilla on mahdollista laskea tärkeysarvot piirteille, jotka maksimoivat ennustustarkkuuden. Tällaisia malleja ovat esimerkiksi satunnaismetsä (engl. random forest) ja "extremely randomized trees". Nämä kaksi mallia valittiin mukaan analyyseihin. Ennustemalleissa regressio kuvaa ennustetta jatkuvana muuttujana, kun taas luokittelu jakaa aineiston ennalta määrättyyn määrään eri kategorioita eli luokkia. 

Satunnaismetsä on yhdistelmämalli, jossa yhdistetään useita päätöspuita (Breiman 2001). Luokittelutehtävässä metsän ulostuloarvoksi valitaan enemmistötulos satunnaispuiden luokittelutuloksista. Regressiotehtävässä ulostuloarvoksi määräytyy satunnaispuiden ennusteiden keskiarvo. Satunnaismetsän ominaisuuksiin kuulu, että eri sisääntulomuuttujille voidaan laskea tärkeysarvot pohjautuen malliin toteutettuun permutaatiotestaukseen (Breiman 2001). Satunnaismetsässä sisääntulodata jaetaan osiin ja yksittäisen päätöspuun ennuste perustuu ositettuun dataan. Jokaisen puun haarassa suoritetaan datan jaottelua perustuen osittain satunnaiseen näytevektorien piirteytyksiin. Extremely randomized trees -malli on samankaltainen kuin satunnaismetsä. Erona mallien välillä on datan osittaminen päätöspuille ja piirteiden jakaminen puun solmuille. Extremely randomized trees -mallissa koko aineisto ositetaan päätöspuille ja piirteiden jako solmuille suoritetaan täysin satunnaisesti (Geurts 2006). Molemmissa malleissa valitaan lopuksi paras piirteiden kombinaatio kullekin päätöspuulle.   

Tässä tutkimuksessa koneoppimismallit opetettiin käyttämällä ennustavina piirteinä psykologisten muuttujien muutoksia 0–36 kk mittausten välillä. Muutokset laskettiin alku- ja loppumittauksen erotuksena siten, että negatiiviset arvot kuvasivat muuttujien pistemäärän laskua ja positiiviset arvot kuvasivat pistemäärän nousua seuranta-ajan alusta sen loppuun. Pisteiden muutokset laskettiin sekä yksittäisille muuttujien väittämille että neljän kategorian summamuuttujille. 

Ennustettavana muuttujana käytettiin painon muutosta joko luokiteltuna (1 = paino laskee seuranta-aikana ≥2,5 % lähtöpainosta ja 0 = paino laskee <2,5 % lähtöpainosta) tai jatkuvana muuttujana (muutos prosentteina 0–36 kk mittausjaksolta). Luokittelun raja-arvoksi valittiin 2,5 %:n painonpudotus, sillä jo tämän suuruisella liikapainon vähentämisellä on osoitettu olevan diabetesriskiä 20–30 prosenttia vähentävä vaikutus (Rintamäki ym. 2020).


##### Tärkeysarvojen laskennat sisälsivät seuraavat vaiheet:
1.	Osallistujien rajaus niihin henkilöihin, jotka olivat mukana koko Muutosmatka-intervention ajan eli 36 kk. Osallistujamäärä rajautui 78 osallistujaan.     
2.	Suhteellisten painon muutosten laskennat intervention alusta intervention loppuun.  
3.	Psykologisten pisteytyksien muutoksien laskennat yksittäisille psykologisille muuttujille ja neljän pääkategorian summamuuttujille intervention alusta intervention loppuun.  
4.	Puuttuvien arvojen käsittely. Summamuuttujien tapauksessa käytettiin mediaani-imputointia ja yksittäisten muuttujien tapauksessa käytettiin 10-lähimmän naapurin imputointia.   
5.	Muuttujat skaalattiin nollakeskiarvoiseksi ja yksikköhajonnalle z-score muunnoksella.  
6.	Verkkohakua käytettiin mallien parametrien optimointeihin.   
7.	Työssä käytettiin viisinkertaista ristiinvalidointia mallien opetuksiin ja testauksiin. Tärkeysarvoista otettiin keskiarvot jakamalla tärkeysarvojen summat viidellä.   
8.	Käytettyihin malleihin liittyy mallien satunnaisuuteen perustuvaa laskennallista epävarmuutta. Tämän vuoksi Vaihe 7 toistettiin yhteensä 100 kertaa. Toistoista laskettiin keskiarvot, keskihajonnat ja mediaaniarvot.      

#### Tulokset
Tuloslistauksessa on huomioitu vain psykologiset summamuuttujat. Yksittäisten muuttujien tärkeysarvojen listaukset on esitetty tiedostossa *Tarkeysarvot.xlsx*. 

#### Luokittelumallit
Käytettyjen satunnaismetsä ja extremely randomized trees mallien luokittelutarkkuudet olivat psykologisille summamuuttujille 60,58 % ja 60,61 %. Tarkkuudet ovat melko heikkoja ja tämän vuoksi tärkeysarvojen tuloksiin täytyy suhtautua varauksella. Taulukossa 1 ja Taulukossa 2 on listattu tärkeysarvojen tulokset luokittelumalleilla. Molemmilla malleilla psykologiset muuttujat listautuivat samaan järjestykseen ja WBSI-summamuuttuja sai suurimman tärkeysarvon.    
  
Taulukko 1. Summamuuttujien tärkeysarvojen tulokset satunnaismetsäluokittelumallilla  
--- |  | Keskiarvo | Keskihajonta |  
--- | --- | --- | --- |  
1 | WBSI | 34,24 % | 0,81 % |    
2 | AAQ-II | 24,74 % | 0,59 % |    
3 | DASS-21 | 22,96 % | 0,64 % |   
4 | GSE | 18,06 % | 0,52 % |    
  
Taulukko 2. Summamuuttujien tärkeysarvojen tulokset extremely randomized trees -luokittelumallilla.  
---  |  | Keskiarvo | Keskihajonta |  
--- | --- | --- | --- |  
1 | WBSI | 29,70 % | 0,47 % |   
2 | AAQ-II | 25,38 % | 0,47 %  |    
3 | DASS-21 | 24,42 %  | 0,49 % |   
4 | GSE | 20,50 % | 0,44 % |    
  
#### Regressiomallit
Regressiomallien tapauksessa ennustettujen arvojen ja todellisten arvojen välille laskettiin Pearsonin korrelaatiokertoimet. Korrelaatiot olivat heikkoja, jonka vuoksi myös regressiomallien tärkeysarvotuloksiin täytyy suhtautua varauksella. Satunnaismetsämallilla korrelaatiokerroin oli *r =* 0,225 ja extremely randomized trees -mallilla korrelaatiokerroin oli *r =* 0,275. Taulukossa 3 ja Taulukossa 4 on listattu tärkeysarvojen tulokset regressiomalleilla. Selvästi malleilla psykologiset summamuuttujat listautuivat eri järjestykseen, joka edelleen heikentää tulosten luetettavuutta.

Taulukko 3. Summamuuttujien tärkeysarvojen tulokset satunnaismetsäregressiomallilla.  
---  |  | Keskiarvo | Keskihajonta |   
--- | --- | --- | --- |  
1 | DASS-21 | 29,49 % | 0,77 % |   
2 | WBSI | 27,22 % | 0,95 % |   
3 | AAQ-II | 23,59 % | 0,79 % |   
4 | GSE | 19,71 % | 0,64 % |   

Taulukko 4. Summamuuttujien tärkeysarvojen tulokset extremely randomized trees -regressiomallilla.  
---  |  | Keskiarvo | Keskihajonta |   
--- | --- | --- | --- |  
1 | DASS-21 | 27,78 % | 0,59 % |   
2 | AAQ-II | 25,05 % | 0,53 % |   
3 | GSE | 24,05 % | 0,55 % |   
4 | WBSI | 23,12 % | 0,54 % |   

#### Pohdintoja
Kokeissa käytetyt koneoppimismallit sisälsivät runsaasti säädettäviä parametreja. Tällaisia parametreja olivat: puiden lukumäärä, puun solmujen laatuun käytetty virhekriteeri, maksimisyvyys puussa, minimi näytteiden määrä puun solmuissa, minimi näytteiden määrä lehtisolmuissa, minimi painokerroin lehtisolmuissa, maksimi piirteiden määrä puun solmuissa, maksimi lehtisolmujen määrä, jne. Kaikki nämä arvot voivat vaikuttaa mallien suorituskykyyn. Verkkohaulla on mahdollista pyrkiä löytämään paras parametrien yhdistelmä liittyen mallien suorituskykyyn. Verkkohaku vaatii kuitenkin hyviä alkuarvauksia muuttujille.  Parametrien säätäminen on työlästä ja vaatii erittäin hyvää mallien tuntemusta. Tämän projektin aikataulun puitteissa ei keretty syventymään riittävästi parametrien valintaan, jonka vuoksi parasta mahdollista suorituskykyä ei välttämättä saavutettu ennustustehtävissä.

Tulevaisuudessa piirteiden valintaan olisi mahdollista kokeilla käyttää yksinkertaisempia malleja, kuten lasso-regressiota (Tibshirani 1996). Julkaisussa (Linja ym. in press) esitetään vertailu, joka perustuu jo aikaisemmin tehtyihin vertailuihin liittyen piirteiden tärkeysarvojen laskentamalleihin. Etäisyyspohjaiset mallit, kuten minimal learning machine (Hämäläinen ym. 2020), ovat toimineet vertailussa hyvin ja ne sisältävät vain yhden säädettävän parametrin, joka on laskennassa käytetty referenssipisteiden lukumäärä.

#### Lähteitä:

Breiman, Random Forests, Machine Learning, 45(1), 5–32, 2001.   
  
Geurts, P., Ernst, D., & Wehenkel, L. (2006). Extremely randomized trees. Machine learning, 63(1), 3-42.  

Hämäläinen, J., Alencar, A. S. C., Kärkkäinen, T., Mattos, C. L. C., Souza Junior, A. H., & Gomes, J. P. P. (2020). Minimal learning machine: Theoretical results and clustering-based reference point selection, Journal of Machine Learning Research 21, pp. 1–29.

Rintamäki, R., Rautio, N., Peltonen, M., Jokelainen, J., Keinänen-Kiukaanniemi, S., Oksa, H., ..., Moilanen, L. (2021). Long-term outcomes of lifestyle intervention to prevent type 2 diabetes in people at high risk in primary health care. Primary care diabetes, 15(3), 444-450. 

Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. Journal of the Royal Statistical Society. Series B (methodological). Wiley. 58 (1): 267–88. 

