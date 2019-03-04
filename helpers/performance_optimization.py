from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split, StratifiedShuffleSplit
from sklearn import metrics
import pandas as pd
import numpy as np
import pdb

###################
## Model Evaluation
###################
def validate_best_model(best_rf, X_test, y_test):
    '''
   Validate fitted model on test data. 

    Params
    ------
    best_rf : object
        fitted/trained scikit-compatible model
    X_test : np.array
        observations x feature matrix for testing
    y_test : vector
        labels corresponding to each row of X_test
    
    Returns
    -------
    metrics_results : dict
        evalaution_metrics: value pair 

    metrics_df : pandas.DataFrame
        metrics_results converted to a dataframe

    model_param : object
        parameters of the scikit-compatible model

    '''
    print("Validating best model on held out set...")
    # use best model to predict on test set
    model_params = best_rf.get_params()
    y_pred = best_rf.predict(X_test)
    #y_proba = best_rf.predict_proba(X_test)

    # compute metrics
    metric_results = regression_metrics(y_test, y_pred)
#    classifier_results = #TODO: Add ability to detect and get classifier results if not regressor
    metrics_df = metrics_to_df('test', metric_results, 'regressor')
    metrics_df.drop(['cv_iter'], axis=1, inplace=True)

    return metric_results, metrics_df, model_params, y_pred

def regression_metrics(y_true, y_pred):
    ''' calcualte evaluation metrics for y_true, y_pred '''

#    pr_score = metrics.precision_score(y_true, y_pred)
#    rc_score = metrics.recall_score(y_true, y_pred)
#    f1_score = metrics.f1_score(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    explained_variance = metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error = metrics.mean_absolute_error(y_true, y_pred)
    mean_squared_error = metrics.mean_squared_error(y_true, y_pred)
    

    # pr curve
#    pr_curve, rc_curve, _ = metrics.precision_recall_curve(y_true, probas_)
#    avg_prec = metrics.average_precision_score(y_true, probas_)

    # roc curve
#    fpr, tpr, _ = metrics.roc_curve(y_true, probas_)
#    auc_trap = metrics.auc(fpr, tpr)  # trap rule area calc
    # roc_auc_default = metrics.roc_auc_score(y_true, probas_)

    # confusion matrix
#    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()

    # brier_score
#    brier_score = metrics.brier_score_loss(y_true, probas_, sample_weight=None, pos_label=1)

#    results_dict = {'pr_score': pr_score, 'rc_score': rc_score, 'f1_score': f1_score,
#                    'fpr': fpr, 'tpr': tpr, 'pr_curve': pr_curve, 'rc_curve': rc_curve, 'avg_prec': avg_prec,
#                    'roc_auc': auc_trap,
#                    'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
#                    'brier_score': brier_score}

    results_dict = {'r2': r2, 'explained_variance': explained_variance, 'mean_absolute_error': mean_absolute_error,
                    'mean_squared_error': mean_squared_error, 'total_count': len(y_pred)}

    return results_dict
    
def metrics_to_df(cv_iter, results_dict, modeltype = 'regressor'):
    '''
    convert results dictionary to dataframe
    '''

    if modeltype == 'regressor':
        temp_df = pd.DataFrame({k: [v] for k,v in results_dict.items()})
        temp_df['cv_iter'] = [cv_iter]

    else:
        total_n = results_dict['tn'] + results_dict['fp'] + results_dict['fn'] + results_dict['tp']
        temp_df = pd.DataFrame({
        'cv_iter': [cv_iter],
        'precision': results_dict['pr_score'],
        'recall': results_dict['rc_score'],
        'f1': results_dict['f1_score'],
        'roc_auc': results_dict['roc_auc'],
        'avg_pr': results_dict['avg_prec'],
        'tn_count': results_dict['tn'],
        'fp_count': results_dict['fp'],
        'fn_count': results_dict['fn'],
        'tp_count': results_dict['tp'],
        'brier_score': results_dict['brier_score'],
        'total_count': total_n})

    return temp_df

#########################
## Parameter optimization
#########################

def grid_search(estimator, param_grid, X_train, y_train, scoring='r2', cv=5, verbose=1):

    print("Tuning hyperparameters...")

    # set up gridsearchCV object
    searchObject = GridSearchCV(estimator=estimator, param_grid=param_grid, scoring=scoring, error_score = 'raise',
                                cv=cv, iid=False, n_jobs=-1, verbose=verbose, return_train_score=True)

    # Fit the grid search to the data
    searchObject.fit(X_train, y_train)

    # organize outputs 
    best_summary = searchObject.cv_results_
    best_param = searchObject.best_params_
    best_rf = searchObject.best_estimator_

    summary_df = pd.DataFrame(best_summary)
    summary_df.drop(['params'], axis=1, inplace=True)

    cols_to_keep = ['mean_fit_time', 'mean_score_time', 'param_max_depth', 'param_max_features', 'param_n_estimators','rank_test_score']
    keep_cols_test = [x for x in summary_df.columns if (x.find('test') != -1)] + cols_to_keep
    keep_cols_train = [x for x in summary_df.columns if (x.find('train') != -1)] + cols_to_keep

    tune_hp_df_test = summary_df.loc[:, keep_cols_test].copy()
    tune_hp_df_train = summary_df.loc[:, keep_cols_train].copy()

    return best_rf, best_param, tune_hp_df_train, tune_hp_df_test

def rand_grid_search(estimator, param_grid, X_train, y_train, num_iters=100, scoring='r2', cv=5, verbose=1):
    '''
    Implements RandomizedSearchCV. Inputs are directly passed to RadomizedSearch CV
    '''

    print("Tuning hyperparameters...")

    # set up RandomizedSearchCV object
    searchObject = RandomizedSearchCV(estimator=estimator, param_distributions=param_grid, n_iter=num_iters, scoring=scoring,
                                      cv=cv, iid=False, n_jobs=-1, verbose=verbose, return_train_score=True, error_score = 'raise')#  random_state=32)

    # Fit the grid search to the data
    searchObject.fit(X_train, y_train)

    best_summary = searchObject.cv_results_
    best_param = searchObject.best_params_
    best_rf = searchObject.best_estimator_

    cv_results = pd.DataFrame(best_summary)
    cv_results.drop(['params'], axis=1, inplace=True)

    # split train and test into 2 dataframes
    time_cols = ['mean_fit_time', 'mean_score_time']
    cols_to_keep = [x for x in cv_results.columns if (x.find('param') != -1)] + time_cols

    keep_cols_test = [x for x in cv_results.columns if (x.find('test') != -1)] + cols_to_keep
    keep_cols_train = [x for x in cv_results.columns if (x.find('train') != -1)] + cols_to_keep
    tune_hp_df_test = cv_results.loc[:, keep_cols_test].copy()
    tune_hp_df_train = cv_results.loc[:, keep_cols_train].copy()

    return best_rf, best_param, tune_hp_df_train, tune_hp_df_test

###############################
## Feature importance functions
###############################

def get_feature_importance(fitted_model, data_df, cols_to_drop=['sample', 'label']):
    '''
    Given a trained model, return a dataframe with feature importance (gini) and std. 
        * assumes that model parameters have already been initialized. 
        * if xgboost model is used, no std is calculated
      
    INPUTS: 
        - fitted_model: 
            model fit to training data (sci-kit model)
            * assumes parameters have already been set.
        
        - data_df: 
            dataframe of the input data used to tune/train/test model 
            * requires that all columns are features except for 'sample' and 'label'
        
        - cols_to_drop: 
            columns names in data_df to remove from dataframe (columns that are not features)

    OUTPUT:
        - dataframe with one row per feature with importance and std columns 

    '''
    #pdb.set_trace()
    data_df = data_df.copy() # to ensure any changes to df are not passsed on 

    # calc importance
    feat_importance = fitted_model.feature_importances_

    try:
        feat_std = np.std([tree.feature_importances_ for tree in fitted_model.estimators_],
                      axis=0)
    except AttributeError:
        feat_std = np.zeros(len(feat_importance))

    sorted_indices = np.argsort(feat_importance)[::-1]
    sorted_feat_importance = feat_importance[sorted_indices]
    sorted_feat_std = feat_std[sorted_indices]

    # feature labels
    data_df.drop(cols_to_drop, axis=1, inplace=True)
    feat_labels = data_df.columns
    sorted_feat_labels = feat_labels[sorted_indices]

    feat_df = pd.DataFrame({'feature': sorted_feat_labels,
                            'importance': sorted_feat_importance,
                            'std': sorted_feat_std},  dtype='object')

    return feat_df       
