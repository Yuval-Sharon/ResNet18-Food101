# ResNet18-Food101

A demo of the results can be found under results_demo.ipynb

The model directory contains the trained model.

The data directory contains the dataset.

The stats directory contains the training and test loss and accuracy.

Main file for training is model_food101.py

it can be run by:

```shell
python model_food101.py --noise 0.2
```

and the statistics can be found in `stats/epoch_stats_<DATE>_noise_0.2.csv`

the trained model can be found in `model/food101_<DATE>_after_train_noise_0.2.pth`
