from sklearn import metrics
import pandas as pd
import numpy as np
from functools import reduce
#import pdb

#######################
#######################
####               ####
#### Random Forest ####
####               ####
#######################
#######################

############
###### Misc:
############

def metrics_to_df(cv_iter, results_dict, dfcols):
    temp_df = pd.DataFrame({x:[results_dict[x]] for x in dfcols})
    if 'tn' in dfcols:
        total_n = results_dict['tn'] + results_dict['fp'] + results_dict['fn'] + results_dict['tp']               
        temp_df['total_n'] = total_n
    temp_df['cv_iter'] = [cv_iter]
    return temp_df

def get_feature_importance(rf_model, X_mat):
    features = [x for x in list(X_mat) if x!="label"]
    feat_importance = rf_model.feature_importances_
#    sorted_indices = np.argsort(feat_importance)[::-1]
#    sorted_
    return pd.DataFrame({'feature': features,
                         'importance': feat_importance})

def average_feature_importance(feat_importances):
    ''' Takes a list of feature importance dfs and returns mean and std feature importances '''
    assert len(feat_importances) > 0, "Attempted to average an empty list of feature importances"
    for i,x in enumerate(feat_importances):
        x.rename(columns={'importance' : 'cv_{}_importance'.format(i)},inplace=True)
    all_feat = reduce(lambda x,y: pd.merge(x,y, on='feature'), feat_importances)
    all_feat['average'] = all_feat[[x for x in list(all_feat) if x!='feature']].mean(axis=1)
    all_feat['std'] = all_feat[[x for x in list(all_feat) if x!='feature']].std(axis=1)
    all_feat.sort_values('average',ascending=False,inplace=True)
    return all_feat
            
##################
###### Classifier:
##################

def evaluate_classifier(y_true, y_pred, probas):
    ''' calculate evaluation metrics for y_true, y_pred, and probas_ (probability of the positive class) '''
    pr_score = metrics.precision_score(y_true, y_pred)
    rc_score = metrics.recall_score(y_true, y_pred)
    f1_score = metrics.f1_score(y_true, y_pred)

    # pr curve
    pr_curve, rc_curve, _ = metrics.precision_recall_curve(y_true, probas)
    avg_prec = metrics.average_precision_score(y_true, probas)

    # roc curve
    fpr, tpr, _ = metrics.roc_curve(y_true, probas)
    auc_trap = metrics.auc(fpr, tpr)  # trap rule area calc
    # roc_auc_default = metrics.roc_auc_score(y_true, probas_)

    # confusion matrix
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()

    # brier_score
    brier_score = metrics.brier_score_loss(y_true, probas, sample_weight=None, pos_label=1)

    results_dict = {'pr_score': pr_score, 'rc_score': rc_score, 'f1_score': f1_score,
                    'fpr': fpr, 'tpr': tpr, 'precision': pr_curve, 'recall': rc_curve, 'avg_pr': avg_prec,
                    'roc': auc_trap,
                    'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
                    'brier': brier_score}

    return results_dict

#################
###### Regressor:
#################

def evaluate_regressor(y_true, y_pred):
    r2 = metrics.r2_score(y_true, y_pred)
    explained_variance = metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error = metrics.mean_absolute_error(y_true, y_pred)
    mean_squared_error = metrics.mean_squared_error(y_true, y_pred)
    return {
            'r2': r2,
            'explained_variance': explained_variance,
            'mae': mean_absolute_error,
            'mse': mean_squared_error
            }

