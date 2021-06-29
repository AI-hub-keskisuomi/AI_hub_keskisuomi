# Sääriluun eminentia-alueen reunantunnistusmenetelmä
Eminentia-alueen reunantunnistusmenetelmä DICOM-formaatin rtg-kuvista: etsii ja palauttaa luun muodon polvinivelessä, keskittyy eminentioiden reunan tunnistamiseen.
### Käyttö: 
* DICOM-kuvakansiopolku funktiolle find_eminentia_edge(path).
  - path muodossa polku/potilasid/kuvaid
  - oaipurkaja.py ja oaipurkaja2.py pitäisi tuottaa oikeanlaisen hakemistorakenteen pakatuista OAI-kuvista.
* Tulos: oikean polven (kuvassa vasemmalla) nivelrakoalue, johon piirretty luiden reunat. Palauttaa listat seuraavista muuttujista:
  - *knee_area*: rajattu polvinivelkuva
  - *tck*: splinifunktio joka myötäilee tibian löytynyttä reunaa (ks. scipy:n interpolate.splprep)
  - *splineimg*: binäärikuva sääriluun löytynyttä reunaa myötäilevästä splinistä (yhtenäinen)
  - *joined_tibia_line*: binäärikuva tibian löytyneestä reunasta ennen splinisovitusta (saattaa katkeilla)
  - *femur_edge*: binäärikuva femurin löytyneestä reunasta

Vaaditut paketit: numpy, matplotlib, scipy, skimage, pydicom, sklearn
	
Menetelmän lyhyt kuvaus:
1. eristä polvialue (funktio find_knee_area)
    1. etsi luut (pääasiassa funktio find_femur)
    2. etsi nivelrako (pääasiassa funktio find_joint_space)
1. eristä luun reuna polvialueesta (loput)
    1. etsi femurin reuna
    2. etsi tibian reuna

2.ii: tibian reunan etsiminen on Canny-pohjainen. Paikoin löytynyt reuna saattaa katketa, koska kuvassa ei ole (tarpeeksi suurta) gradienttia. Katkeamisia tapahtuu usein. Splini on jatkuva ja sileä koko matkalta eli yhdistää reunan osat.

## Menetelmästä
Nivelrikko on yleinen (maailman yleisin nivelsairaus) ja kallis sairaus. Sen kustannukset Suomessa on arvioitu olevan miljardin luokkaa vuodessa. Sen aikainen tunnistaminen säästäisi rahaa ja kipua. Nivelrikon diagnosoinnissa käytetään röntgenkuvaa ja kliinisiä tutkimuksia. Röntgenkuvassa näkyy sairaudelle tyypillisiä muutoksia nivelen luurakenteessa. Aikainenen tunnistaminen on kuitenkin vaikeaa. Tunnistukseen röntgenkuvista on kehitetty automaattisia menetelmiä, mutta parhaatkaan niistä eivät riittävän luotettavasti ole kyenneet tunnistamaan aikaisen vaiheen nivelrikkoa. Tällä hetkellä parhaat kehitetyt menetelmät ovat konvoluutioneuroverkkoja.

Hypoteesin mukaan eminentioiden, eli sääriluun nivelalueen keskiosan (kuvassa) vuoria muistuttavien muotojen muoto on nivelrikon varhainen merkki. Tämä menetelmä eristää luun reunan keskittyen eminentia-alueeseen.

Menetelmä nojaa suurelta osin Canny-reunantunnistusmenetelmään (J. Canny 1986). 

### 1.1. Luiden reunojen etsintä
1. Haetaan kuvasta pystysuuntaiset reunat (horisontaalinen sobel ja canny)
2. Etsitään keskiarvoltaan kirkkain kahden löytyneen reunan välinen alue
    - Lasketaan riveittän ylhäältä alas
    - Seuraavan rivin tulosta painotetaan edellisen rivin tuloksella
    - Paluuarvona on binäärikuva luualueesta, jossa luualue = 1 ja muu alue = 0

### 1.2. Nivelalueen etsintä
1. Lineaariregressio luuta pitkin (pystysuuntainen regressio eli x- ja y-akseli vaihdetaan keskenään)
2. Silmukassa
    - Regressiosuoran suuntainen kuvan derivaatta f
    - Haetaan f:n suurin vaihtelu max(f(i) - f(j)), missä arvojen etäisyys on määriteltyä pienempi dist(f(i), f(j)) < a
    - Jos f:ssä on lokaaleja minimejä tai maksimeja, joiden itseisarvo > b, f rajataan lähimpänä 0:aa oleviin ääriarvoihinsa
    - Femurin alustuspiste ko. suoralla on f(j)
    - siirretään regressiosuoraa pykälällä sivulle ja palataan silmukan alkuun
3. rajataan nivelalue alustuspisteiden avulla
- Paluuarvona rajattu nivelalue (= *knee_area*) sekä binäärikuva femurin alustusreunasta

### 2.1. Femurin reuna
- käyttää kuvana rajattua nivelaluetta
1. Reunaversio kuvasta Canny-algoritmilla (herkillä asetuksilla) = canny-kuva
2. Poimitaan kandidaattireunoiksi ne canny-kuvan reunat, jotka menevät ovat (edes osittain) päällekäisiä alustusreunan kanssa.
3. Jaetaan kandidaattireunat erillisiin haaroihin
3. Poimitaan sarakkeittain alimman haaran reunapisteet
- Tuloksena on binäärikuva Femurin reunasta (kuvassa sinisellä) (= *femur_edge*)

### 2.2. Tibian reuna
1. Vaakasuuntaiset reunat
    - Löytyvät kertomalla canny-kuva ja horisontaalinen gradienttiversio kuvasta (sobel) keskenään
    - Reunapikseleistä valitaan sarakkeittain ylimmät, jotka ovat reisiluun reunaa alempana
2. Virheellisten reunojen poisto
    - Poimitaan kaikki canny-reunat, jotka menevät (ainakin osittain) päällekäin 1. kohdan vaakasuuntaisten reunojen kanssa
    - Jaetaan erillisiin haaroihin
    - Yhdistetään haarat lineaaripidennyksillä, jotka sovitetaan reunojen päihin.
        - Yhdistämistä varten reunapisteet pitää järjestää
    - Valitaan ylimmät reunat 
    - Valitaan ylimmistä reunoista sellaiset reunapisteet, jotka ovat myös canny-kuvassa
3. Haetaan loput Cannylla löytyneet oikeat reunat
    - Haetaan erilliset reunapätkät
    - Haetaan sellaiset canny-reunapikselit, jotka 
        - yhdistävät kaksi reunaa tai
        - lähestyvät toisen reunan oikeasta päätepisteetä kohti toisen reunan vasenta päätepistettä
    - Järjestetään reunapisteet (= *joined_tibia_line*)
    - Sovitetaan splini (= *tck*)
    - piirretään splinibinääriuva (= *splineimg*)
	
Huomioita
* Koodissa on paikoin epäsäännöllisyyksiä johtuen sen luomistavasta, esim. Cannyn parametrit on eri tavalla mukana eri funktioissa
* Koodissa saattaa olla myös joitain turhaksi jääneitä rivejä ja funktioitakin

Ongelmia
- Reunan järjestäminen ei toimi aina. Sen johdosta splini menee sotkuun.
    - reunan järjestämisen toimintaperiaate:
        - ohennetaan reuna skimagen skeletonize-funktiolla
        - etsitään ja järjestetään (vasen -> oikea) reunan päät sitä varten luodulla filtterillä head
        - aloittamalla vasemmasta päästä etsitään aina edellistä lähin reunapikseli 
- Skeletonize ei muutenkaan aina käyttäydy halutulla tavalla.
- Sääriluun reuna "tarraa" reisiluuhun eli löytää saman reunan.
    - jos reisiluun reunaa ei löydy kaikista paikoista, voinee tulla ongelma
    - lisäksi ongelma on ilmeisesti ainakin funktiossa initialize_tibia_line2
