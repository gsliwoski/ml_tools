## Simple scripts to automate some machine learning tasks and generate custom features

### Current models

1. Random Forest - Classifier (tested) and Regressor (untested) in sklearn - Trains model using default params or indicated param set and plots and records performance metrics and feature importances

### Current features

1. Variant neighborhood in 3D protein structure

### Params set if by using default parameters

Note: If supplying a params file, any of these parameters not included will use default here

#### For cross-validation:

- n_splits (default = 10)
- test_size (default = 0.2)
- train_size (default = 0.8)
- random_state (default = 32)

#### For random forest classifier

- max_depth (default = 10)
- class_weight (default = 'balanced')
- n_estimators (default = 10)

#### For random forest regressor

- max_depth (default = 10)

#### For ANN classifier

- hidden_layer_sizes (default = (100,50))
- solver (default = 'lbfgs')

#### For ANN regressor

- hidden_layer_sizes (default = (100,50))
- solver (default = 'lbfgs')
