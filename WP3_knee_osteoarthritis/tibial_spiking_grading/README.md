# Tibial spiking grading

Project files for developing a model for automatically assessing tibial spiking from radiographs.

The model is based on ImageNet pre-trained ResNeXt50-32x4d. We utilized the
pre-trained weights for every layer, only the classification head was replaced
with fully connected 2048x2 layers.

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

