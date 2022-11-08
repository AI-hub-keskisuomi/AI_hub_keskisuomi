## Psykologisten tekijöiden vaikuttavuuden arviointi Muutosmatka-interventiossa

Psykologiset tekijät voidaan jakaa neljään kategoriaan:
- Psykologinen joustavuus (AAQ-II)
- Ajattelutoiminnot (WBSI)
- Minäpystyvyys (GSE)
- Mieliala (DASS-21)

Potilaille oli mahdollista laskea psykologisten tekijöiden pisteytykset kyselylomakekyselyiden perusteella. Alhaiset arvot AAQ, WBSI ja DASS pisteytyksissä ja korkeat arvot GSE-pistetyksessä kuvaavat hyviä tuloksia. Osalla muuttujista on olemassa myös alakategorioita. 

Terveyteen liittyväksi riskitekijäksi valittiin kohdehenkilöiden painotiedot ja psykologisten tekijöitä verrattiin painotietoihin. Tavoitteena oli pyrkiä tunnistamaan ne psykologiset tekijät, jotka vaikuttavat eniten painon muutoksiin. Tuloksia oli mahdollista tarkastella tilastollisesti korrelaatioanalyyseillä ja koneoppimislähtöisesti piirteiden tärkeysarvojen tarkasteluilla. 

#### Piirteiden tärkeysarvojen tarkastelu

Tietyillä luokittelu- ja regressiomalleilla on mahdollista laskea tärkeysarvot piirteille, jotka maksimoivat ennustustarkkuuden. Tällaisia malleja ovat esimerkiksi satunnaismetsä (engl. random forest) ja "extremely randomized trees". Nämä kaksi mallia valittiin mukaan analyyseihin. Ennustemalleissa regressio kuvaa ennustetta jatkuvana muuttujana, kun taas luokittelu jakaa aineiston ennalta määrättyyn määrään eri kategorioita eli luokkia. 

Satunnaismetsä on yhdistelmämalli, jossa yhdistetään useita päätöspuita (Breiman 2001). Luokittelutehtävässä metsän ulostuloarvoksi valitaan enemmistötulos satunnaispuiden luokittelutuloksista. Regressiotehtävässä ulostuloarvoksi määräytyy satunnaispuiden ennusteiden keskiarvo. Satunnaismetsän ominaisuuksiin kuulu, että eri sisääntulomuuttujille voidaan laskea tärkeysarvot pohjautuen malliin toteutettuun permutaatiotestaukseen (Breiman 2001). Satunnaismetsässä sisääntulodata jaetaan osiin ja yksittäisen päätöspuun ennuste perustuu ositettuun dataan. Jokaisen puun haarassa suoritetaan datan jaottelua perustuen osittain satunnaiseen näytevektorien piirteytyksiin. Extremely randomized trees -malli on samankaltainen kuin satunnaismetsä. Erona mallien välillä on datan osittaminen päätöspuille ja piirteiden jakaminen puun solmuille. Extremely randomized trees -mallissa koko aineisto ositetaan päätöspuille ja piirteiden jako solmuille suoritetaan täysin satunnaisesti (Geurts 2006). Molemmissa malleissa valitaan lopuksi paras piirteiden kombinaatio kullekin päätöspuulle.   



#### Lähteet:

Breiman, Random Forests, Machine Learning, 45(1), 5–32, 2001.   
  
Geurts, P., Ernst, D., & Wehenkel, L. (2006). Extremely randomized trees. Machine learning, 63(1), 3-42.

