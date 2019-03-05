# This script will train, test and report model evaluation metrics.

import os
import sys
import time
import numpy as np
import pandas as pd
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from datetime import datetime
from scipy.interpolate import interp1d
from sklearn import metrics
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit

#from IPython.core.debugger import set_trace
sys.path.append('/dors/capra_lab/users/sliwosgr/ml_tools/helpers/')
from model_evaluation import *
import estimator_plots
#from performance_optimization import *
import IO

DATE = datetime.now().strftime('%Y-%m-%d')


# -----------
# FUNCTIONS
# -----------

def run_ann(X_mat, y_labels, params, regressor):
    '''
    Run random forest model with CV. If classifier, use StratifiedShuffleSplit, if regressor use ShuffleSplit
    
        Params:
        -------
            X_mat : feature matrix df
            
            y_labels : label matrix df
            
            params : dict of initialized parameters
            
            regressor : is a regressor (T/F)

        Returns:
        --------
                    
    '''
    # Subset parameters to use for split setup
    cv_params = {x[0]:x[1] for x in params.items() if x[0] in ['n_splits','test_size','train_size','random_state']}

    # Run shuffler specific to model type
    if regressor:
        cross_val = ShuffleSplit(**cv_params)
    else:
        cross_val = StratifiedShuffleSplit(**cv_params)

    cross_val.get_n_splits(X_mat, y_labels)
    
    # Subset model-based parameters
    model_paramlist = ['hidden_layer_sizes',
                       'activation',
                       'solver',
                       'alpha',
                       'batch_size',
                       'learning_rate',
                       'learning_rate_init',
                       'power_t',
                       'max_iter',
                       'shuffle',
                       'random_state',
                       'tol',
                       'momentum',
                       'verbose',
                       'warm_start',
                       'nesterovs_momentum',
                       'early_stopping',
                       'validation_fraction',
                       'beta_1',
                       'beta_2',
                       'epsilon',
                       'n_iter_no_change']
    
    model_params = {x[0]:x[1] for x in params.items() if x[0] in model_paramlist}

    # Set up appropriate model                
    if regressor:
        model = MLPRegressor(**model_params)
    else:
        model = MLPClassifier(**model_params)                
        
    # intialize storage variables
    classifier_metrics = ['recall',
                          'precision',
                          'avg_pr',
                          'fpr',
                          'tpr',
                          'roc',
                          'brier']
    regressor_metrics = ['r2',
                         'explained_variance',
                         'mae',
                         'mse']
    prediction_keys = ['test_preds',
                       'train_preds']
    shared_keys = ['cv_test_df',
                   'cv_train_df',
                   'feature_importance']                       

    storage_vars = {x:list() for x in classifier_metrics+regressor_metrics+prediction_keys}
    storage_vars.update({x:pd.DataFrame() for x in shared_keys})
    storage_vars['feature_importance'] = list()

    # Run the models
    for loop_ind, train_ind_test_ind in enumerate(cross_val.split(X_mat, y_labels)):
        print("Training and Cross-Validating Set {} out of {}".format(loop_ind+1, cv_params['n_splits']))
        train_ind, test_ind = train_ind_test_ind[0], train_ind_test_ind[1]

        # train
        model.fit(X_mat.iloc[train_ind], y_labels.iloc[train_ind])

        # predict
        y_pred_test = model.predict(X_mat.iloc[test_ind])
        y_pred_train = model.predict(X_mat.iloc[train_ind])
        
        # Gather appropriate metrics
        if not regressor:
            # Classifier metrics
            probas_test = model.predict_proba(X_mat.iloc[test_ind])[:,1]
            probas_train = model.predict_proba(X_mat.iloc[train_ind])[:,1]
            test_results = evaluate_classifier(y_labels.iloc[test_ind], y_pred_test, probas_test)
            train_results = evaluate_classifier(y_labels.iloc[train_ind], y_pred_train, probas_train)

            # ROC
            storage_vars['roc'].append(test_results['roc'])
            storage_vars['fpr'].append(test_results['fpr'])
            storage_vars['tpr'].append(test_results['tpr'])

            # PR Curve
            storage_vars['recall'].append(test_results['recall'])
            storage_vars['precision'].append(test_results['precision'])
            storage_vars['avg_pr'].append(test_results['avg_pr'])

            # brier_score
            storage_vars['brier'].append(test_results['brier'])

            dfcols = [x for x in classifier_metrics if x not in ['recall','precision','tpr','fpr']]
        else:
            # Regressor metrics
            test_results = evaluate_regressor(y_labels.iloc[test_ind], y_pred_test)
            train_results = evaluate_regressor(y_labels.iloc[train_ind], y_pred_train)
            
            # Metrics
            storage_vars['r2'].append(test_results['r2'])
            storage_vars['explained_variance'].append(test_results['explained_variance'])
            storage_vars['mae'].append(test_results['mae'])
            storage_vars['mse'].append(test_results['mse'])

            dfcols = regressor_metrics
           
        # Record test predictions
        storage_vars['test_preds'].append(pd.DataFrame({
                                          'true': y_labels.iloc[test_ind],
                                          'predicted': y_pred_test}))
        storage_vars['train_preds'].append(pd.DataFrame({
                                           'true': y_labels.iloc[train_ind],
                                           'predicted': y_pred_train}))                                                        


            
        # convert metrics to df and append
        temp_test_df = metrics_to_df(loop_ind, test_results, dfcols)
        temp_train_df = metrics_to_df(loop_ind, train_results, dfcols)
        storage_vars['cv_test_df'] = storage_vars['cv_test_df'].append(temp_test_df)
        storage_vars['cv_train_df'] = storage_vars['cv_train_df'].append(temp_train_df)
#        storage_vars['feature_importance'].append(get_feature_importance(model,X_mat))
#        storage_vars['coefs'].append(model.coefs_)
#        storage_vars['intercepts'].append(model.intercepts_)
    return storage_vars

# -----------
# MAIN
# -----------

if __name__ == "__main__":

    features, labels, parameters, model_type, noopt, outfiles, model_id = IO.initialize('ann')
    
    # load data
    # train and test
    metrics_results = run_ann(features, labels, parameters, model_type=='reg')

    # Write test predictions
    IO.write_test_predictions(metrics_results['test_preds'], outfiles['PREDICTIONS_FILE'])    

    # Plot metrics
    if model_type=='reg':
#        estimator_plots.plot_scatter(y_test, y_pred, model_id, outfiles['SCATTER_PLOT_FILE'])
#        estimator_plots.plot_predictions(y_test, y_pred, model_id, outfiles['PREDS_PLOT_FILE'])
        pass       
    else:
        estimator_plots.plot_roc(metrics_results['fpr'], metrics_results['tpr'], metrics_results['roc'],
                         model_id, roc_fig_file=outfiles['ROC_FIG_FILE'])
        estimator_plots.plot_pr(metrics_results['precision'], metrics_results['recall'], metrics_results['avg_pr'],
                        model_id, pr_fig_file=outfiles['PR_FIG_FILE'])
    #IO.record_performance(metrics_results, model_id, outfiles['PERFORMANCE_FILE'], model_type=='reg')        
    
    # Write metrics
    metrics_results['cv_test_df'].to_csv(outfiles['TEST_METRIC_DF_FILE'],header=True,index=False,sep="\t")
    metrics_results['cv_train_df'].to_csv(outfiles['TRAIN_METRIC_DF_FILE'],header=True,index=False,sep="\t")

    print("Finished ANN.\n Outputs are in {}".format(outfiles['OUTPUT_DIR']))
