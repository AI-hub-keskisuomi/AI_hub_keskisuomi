## Psykologisten tekijöiden vaikuttavuuden arviointi Muutosmatka-interventiossa

#### Piirteiden tärkeysarvojen tarkastelu

Psykologiset tekijät voidaan jakaa neljään kategoriaan:
- Psykologinen joustavuus (AAQ-II)
- Ajattelutoiminnot (WBSI)
- Minäpystyvyys (GSE)
- Mieliala (DASS-21)

Potilaille oli mahdollista laskea psykologisten tekijöiden pisteytykset kyselylomakekyselyiden perusteella. Alhaiset arvot AAQ, WBSI ja DASS pisteytyksissä ja korkeat arvot GSE-pistetyksessä kuvaavat hyviä tuloksia. Osalla muuttujista on olemassa myös alakategorioita. Terveyteen liittyväksi riskitekijäksi valittiin kohdehenkilöiden painotiedot ja psykologisten tekijöitä verrattiin painotietoihin. Tavoitteena oli pyrkiä tunnistamaan ne psykologiset tekijät, jotka vaikuttavat eniten painon muutoksiin. 

##### Koneoppimismallit 

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

#### Luokittelumallit
Käytettyjen satunnaismetsä ja extremely randomized trees mallien luokittelutarkkuudet olivat psykologisille summamuuttujille 60,58 % ja 60,61 %. Tarkkuudet ovat melko heikkoja ja tämän vuoksi tärkeysarvojen tuloksiin täytyy suhtautua varauksella. Taulukossa 1 ja Taulukossa 2 on listattu tärkeysarvojen tulokset luokittelumalleilla. Molemmilla malleilla psykologiset muuttujat listautuivat samaan järjestykseen ja WBSI-summamuuttuja sai suurimman tärkeysarvon.    





#### Lähteet:

Breiman, Random Forests, Machine Learning, 45(1), 5–32, 2001.   
  
Geurts, P., Ernst, D., & Wehenkel, L. (2006). Extremely randomized trees. Machine learning, 63(1), 3-42.

