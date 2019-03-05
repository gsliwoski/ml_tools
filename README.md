## Simple scripts to automate some machine learning tasks and generate custom features

### Current models

1. Random Forest - Classifier (tested) and Regressor (untested) in sklearn - Trains model using default params or indicated param set and plots and records performance metrics and feature importances

### Current features

1. Variant neighborhood in 3D protein structure

### Misc tools

1. process_variants.py = Drops feature columns with 0 variance and highly correlated feature columns (keeps first)
2. filter_corr_stepwise.py = Stepwise correlated column drop to help with memory issues when many features
3. filter_corr_random_chunks.py = Chunk-based correlation column dropping to avoid memory issues, may miss some correlations
4. filter_corr_iterative.py = Very slow stepwise correlation column dropping which will find all with no memory issues but takes forever.

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
