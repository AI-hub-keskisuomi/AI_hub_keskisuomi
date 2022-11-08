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


#### Tulokset


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



#### Future research

Every machine learning models have many parameters to be optimized, and using these optimized values may increase the performance of the models. For example, a Grid search can be used, but it requires good initial guesses because the search combines initial parameters and uses that information in the optimization. The selection of the initial parameters requires a good understanding of the used models, e.g., parameter values can be examined from the machine learning literature. 
