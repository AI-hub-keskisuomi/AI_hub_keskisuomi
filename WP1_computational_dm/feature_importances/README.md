## Psykologisten tekijöiden vaikuttavuuden arviointi Muutosmatka-interventiossa
### Case 1: Piirteiden tärkeysarvojen tarkastelu

Psykologiset tekijät voidaan jakaa neljään kategoriaan:
- Psykologinen joustavuus (AAQ-II)
- Ajattelutoiminnot (WBSI)
- Minäpystyvyys (GSE)
- Mieliala (DASS-21)

Potilaille oli mahdollista laskea psykologisten tekijöiden pisteytykset kyselylomakekyselyiden perusteella. Alhaiset arvot AAQ, WBSI ja DASS pisteytyksissä ja korkeat arvot GSE-pistetyksessä kuvaavat hyviä tuloksia. Osalla muuttujista on olemassa myös alakategorioita. 

Terveyteen liittyväksi riskitekijäksi valittiin kohdehenkilöiden painotiedot ja psykologisten tekijöitä verrattiin painotietoihin. Tavoitteena oli pyrkiä tunnistamaan ne psykologiset tekijät, jotka vaikuttavat eniten painon muutoksiin. Tuloksia oli mahdollista tarkastella tilastollisesti korrelaatioanalyyseillä ja koneoppimislähtöisesti piirteiden tärkeysarvojen tarkasteluilla. Tietyillä luokittelu- ja regressiomalleilla on mahdollista laskea painoarvot piirteille, jotka maksimoivat ennustustarkkuuden. Tällaisia malleja ovat esimerkiksi satunnaismetsä (eng. Random forest) ja "Extremely randomized trees". Nämä kaksi mallia valittiin mukaan analyyseihin.

Tärkeysarvojen laskennat sisälsivät seuraavat vaiheet:
1. Potilaiden rajaus niihin henkilöihin, jotka olivat mukana koko Muutosmatka-intervention ajan eli 36 kk. Potilasmäärä rajautui 78 potilaaseen.  
2. Suhteellisten painon muutosten laskennat intervention alusta intervention loppuun.
3. Psykologisten pisteytyksien muutoksien laskennat yksittäisille psykologisille muuttujille ja neljän pääkategorian summamuuttujille intervention alusta intervention loppuun.
4. Puuttuvien arvojen käsittely. Puuttuvat summamuuttujat imputoitiin mediaani-imputoinnilla ja puuttuvien yksittäisten muuttujien imputoinnissa käytettiin 10-lähimmän naapurin imputointia. 
5. Datan skaalaus nollakeskiarvoiseksi ja yksikköhajonnalle Z-score muunnoksella.
6. Verkkohakua (eng. Grid search) oli mahdollista käyttää mallien parametrien optimointeihin.
7. Koneoppimismalleihin liittyy mallien satunnaisuuteen perustuvaa laskennallista epävakautta. Tämän vuoksi suoritettiin yhteensä 100 uudelleen ajoa. Lisäksi malleihin liittyy riski ylioppia aineisto, jonka vuoksi tulokset voivat vaikuttaa hyviltä vain mikäli opetukseen ja testaukseen käytetään samaa dataa. Ristiinvalidointi on keino ehkäistä ylioppimista. Ideana on pilkkoa data opetus- ja testiaineistoksi määrätty määrä kertoja siten että kukin näyte on testiaineistossa tasan yhden kerran. Työssä käytettiin viisinkertaista ristiinvalidointia, jossa datan jaottelu suoritettiin yhteensä viisi kertaa kullakin ajolla.    
8. Tärkeysarvot keskiarvoistettiin yhdellä ajolla jakamalla viidellä.
9. Tärkeysarvot keskiarvoistettiin sadan uudelleen ajon jälkeen. Lisäksi arvoille laskettiin keskihajonnat ja mediaanit.    

Saadut tulokset perustuivat regressioennusteisiin siten, että painon suhteellista muutosta ennustettiin jatkuvana muuttujana. Lisäksi mallien suorituskykyä tarkasteltiin luokittelussa siten, että ennustettavana muuttujana oli luokkatieto painon suhteellisesta muutoksesta (0 = painon suhteellinen muutos < 2,5 % ja 1 = painon suhteellinen muutos >= 2,5 %). 

#### Tulokset

Tulosten todenperäisyyttä oli mahdollista arvioida käyttämällä kahta eri mallia ja vertailla mallien tuloksia. Toinen lähestymistapa tulosten arviointeihin oli laskea mallien ennustustarkkuudet sekä luokittelu- että regressiotehtävässä. Keskimääräiset ennustustarkkuudet olivat 61 % sekä satunnaismetsäluokittelumallilla että Extra Tree -luokittelumallilla, kun piirteinä käytettiin psykologisten summamuuttujien muutoksia. Samoilla piirteillä satunnaismetsäregressiomalli ja Extra Tree -regressiomalli antoivat ennustettujen ja todellisten suhteellisten painoarvon muutoksien välisiksi korrelaatiokertoimiksi r = 0,225 ja r = 0,275. Korrelaatiokertoimet ovat selkeästi alhaiset, joten regressiomalleilla saatuihin tuloksiin täytyy suhtautua varauksin. Luokittelumallien tulokset ovat samankaltaisia piirteiden tärkeysarvojen osalta ja näin ollen tulokset ovat hyvin suuntaa antavia. Keskimääräiset luokittelutarkkuudet ovat tosin hieman heikkoja.   

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
1 | DASS | 0,295 | (0,008) | 0,295 | 
2 | WBSI | 0,272 | (0,010) | 0,272 | 
3 | AAQ | 0,236 | (0,008) | 0,236 | 
4 | GSE | 0,197 | (0,006) | 0,196 | 

Summamuuttujien tärkeysarvojen tulokset Extra Tree -regressiomallilla:
---  |  | Mean | (Std) | Median | 
--- | --- | --- | --- |--- |
1 | DASS | 0,278 | (0,006) | 0,278 | 
2 | AAQ | 0,250 | (0,005) | 0,251 | 
3 | GSE | 0,241 | (0,005) | 0,241 | 
4 | WBSI | 0,231 | (0,005) | 0,231 | 

Tärkeysarvotulokset yksittäisille psykologisille muuttujille ja WBSI-muuttujan alakategorioille ovat tiedostossa: 'Tarkeysarvojen_tuloksia.xlsx'.

#### Jatkokehitys

Kaikilla neljällä koneoppimismallilla on suuri määrä optimoitavia parametreja, joita muokkaamalla on mahdollisuus pyrkiä lisäämään ennustustarkkuuksia hyödyntäen esimerkiksi verkkohakua. Mallien parametrit vaativat kuitenkin hyviä alkuarvauksia, koska verkkohaku testaa käyttäjän antamien parametrien kombinaatioita. Hyvien alkuarvauksien valinta vaatii syvällisempää perehtymistä käytettyihin menetelmiin esimerkiksi aiheeseen liittyvän kirjallisuuden kautta. 

## Effectivity analysis of psychological variables in lifestyle intervention  
### Case 1: Using feature importances of classification and regressions models

Psychological variables can be divided into the four categories:
- Psychological flexibility (AAQ-II)
- Thought suppression (WBSI)
- Self-efficacy (GSE)
- Psychological distress (DASS-21)

It was possible to compute points for each category by using questionnaires. Lower values in AAQ, WBSI, and DASS, and higher values in GSE indicated good results, respectively. In addition, some variables consisted of sub-categories.

Patients' weights were selected to risk factors of well-being. Psychological variables were compared to the risk factors. The study aimed to identify the psychological variables which effected most to the weight changes. It was possible to use statitical methods (e.g., methods of correlation analysis) or use feature rankings of machine learning methods. Random forest and Extremely randomized trees methods can be used for computing feature rankings based on predicted outcomes. These models were selected to the analysis.       

The following steps were used for computing feature importances of the selected models:
1. Selection of the target data. Only patients who finished the 36-monthly intervention were accepted. Total number of the patients was 78 after selection.
2. Relative weight changes from the begin of the intervention to the end of the intervention were computed. 
3. Changes in individual psychological variables and in summed psychological variables of the four main categories throughout the 36-montly intervention were computed. 
4. Handling missing values. Missing values in the summed variables were imputed by using a median imputation. Individual psychological variables were imputed by using a 10-nearest neighbors imputation.
5. Z-score scaling of data to the zero mean and unit standard deviation.
6. Parameters of the models were optimized by using a Grid search.
7. Machine learning methods may have some bias caused by random basis models. Therefore, the models were rerun 100 times. In addition, there is a risk that models overfit the data. Cross-validation in effective way to prevent the overfitting. K-fold cross validation splits data K times to the training data and testing data and each sample is tested by once. In the study, 5-fold cross validation was used.  
8. Obtained results was an average of the five test fold.
9. Mean results, standard deviations, and medians were computed over 100 reruns. 

The obtained feature importances were based on regression and classification results. The regression models were used to predict continuous valued weight information over all patients and classification models attempted to predict weight losses as categorical variables (i.e., data vector was labeled to 1 if weight loss was at least 2.5 % and labeled to 0 otherwise).    

#### Results

It is possible to evaluate validity of feature importances by comparing the results obtained from two different models. Another approach is measure actual prediction accuracies in regression and classification tasks. Random forest classifier and Extremely randomized trees classifier both achieved approximately 61 % prediction accuracy when the selected features were changes in the psychological variables (four variables). The models produced the analogical and stable results even the prediction accuracies were not perfect. The correlation coefficients of Random forest regressor and Extremely randomized trees regressor, computed through real weight changes against predicted weight changes, were r = 0,225 and r = 0,275, respectively. The correlation results were quite weak, and therefore, the feature importances obtained by the regression models may not be well accurate. 

Feature importances obtained with Random forest classifier:
---  |  | Mean | (Std) | Median | 
--- | --- | --- | --- |--- |
1 | WBSI | 0,342  | (0,008)  | 0,343  | 
2 | AAQ | 0,247 | (0,006) | 0,248 | 
3 | DASS | 0,230  | (0,006)  | 0,230 | 
4 | GSE | 0,181  | (0,005)  | 0,181 | 

Feature importances obtained with Extra Tree classifier:
---  |  | Mean | (Std) | Median | 
--- | --- | --- | --- |--- |
1 | WBSI | 0,297 | (0,005) | 0,298 | 
2 | AAQ | 0,254 | (0,005)  | 0,254 | 
3 | DASS | 0,244  | (0,005) | 0,244 | 
4 | GSE | 0,205 | (0,004) | 0,204 | 

Feature importances obtained with Random forest regressor:
---  |  | Mean | (Std) | Median | 
--- | --- | --- | --- |--- |
1 | DASS | 0,295 | (0,008) | 0,295 | 
2 | WBSI | 0,272 | (0,010) | 0,272 | 
3 | AAQ | 0,236 | (0,008) | 0,236 | 
4 | GSE | 0,197 | (0,006) | 0,196 | 

Feature importances obtained with Extra Tree regressor:
---  |  | Mean | (Std) | Median | 
--- | --- | --- | --- |--- |
1 | DASS | 0,278 | (0,006) | 0,278 | 
2 | AAQ | 0,250 | (0,005) | 0,251 | 
3 | GSE | 0,241 | (0,005) | 0,241 | 
4 | WBSI | 0,231 | (0,005) | 0,231 | 

The results for individual psychological variables and for sub-categories of WBSI variable are availabe in the file: 'Tarkeysarvojen_tuloksia.xlsx'.





