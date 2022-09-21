# A model for classifying tibial spiking from plain radiographs

Project files for developing a model for automatically assessing tibial spiking from radiographs.

The model is based on ImageNet pre-trained ResNeXt50-32x4d. We utilized the
pre-trained weights for every layer, only the classification head was replaced
with dense layer of 2048 units and 2 outputs.

## Using 

For running a grid search, create a directory containing `params.json` (see directory `example_ex` for an example) and run

```
python grid_search.py -s <experiment_dir> 
```

The results of the grid search with the top model measured by validation accuracy is saved in `<experiment_dir>`.

To evaluate the top model run
```
python test_model.py -s <experiment_dir> 
```
Trained model is located in repository directory `trained_model`.


# Menetelmä eminentian terävöitymisen luokitteluun röntgenkuvista

Tämä repositorio sisältää analyysikoodin jolla malli on koulutettu, valmiiksi koulutetun neuroverkkomallin.

Luokittelumallin soveltaa ImageNet aineistoon sovitettua ResNeXt50-32x4d mallia, josta korvasimme viimeisen varsinaista luokittelun suorittavan osan 2048 neuronilla ja kahdella ulostulolla.

Trained model is located in repository directory `trained_model`.

## Käyttöohjeet

Koulutettua mallia voidaan käyttää komentamalla
```
python test_model.py -s <experiment_dir> 
```
Tutustu `test_model.py` ja `dataset.py` nähdäksesi miten mallia käytetään.


Grid-haku voidaan ajaa hyperparametrien säätämiseksi komennolla 
```
python grid_search.py -s <experiment_dir> 
```
hakemisto `example_ex` sisältää esimerkin miten parametrit määritetään.
