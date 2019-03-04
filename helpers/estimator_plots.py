import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from numpy import polyfit,poly1d
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from sklearn import metrics

##################
## Regressor plots
##################

def plot_scatter(x, y, title, outfile):
    '''
    plot scatter plot of actual vs predicted into png file
    
        INPUTS:
            * actual: list of values
            * predicted: list of values
            * title: identification to prepend to main title
            * outfile: destination file for png           
    '''
    
    main_title = f"{title} scatter plot"
    plt.style.use("seaborn-whitegrid")
    plt.scatter(x,y,marker='o',c='blue')
    plt.xlabel('actual')
    plt.ylabel('predicted')
    plt.title(main_title)
    
    # Add a trendline
    tline = poly1d(polyfit(x,y,1))
    plt.plot(x,tline(x),"r--")
    
    plt.savefig(outfile)    

def plot_predictions(actual,predicted,title,outfile):
    '''
    plot labels comparing actual to predicted into png file
    
        INPUTS:
            * actual: list of values
            * predicted: list of values
            * title: identification to prepend to main title
            * outfile: destination file for png           
    '''
    plt.close()
    main_title = f"{title} label comparison"   
    actual_color = "blue"
    pred_color = "red"
    plt.style.use("seaborn-whitegrid")
    plt.title(main_title)
    
        
    # Sort by actual
    df = pd.DataFrame.from_dict({'actual':actual,'predicted':predicted})
    df.sort_values(by='actual',inplace=True)

    # Set the ylimit
    min_y = df.iloc[0].min()
    max_y = df.iloc[-1].max()
    plt.ylim(bottom=min_y,top=max_y)

    # plot labels       
    plt.plot(df.reset_index().actual, label="actual", color=actual_color)
    plt.plot(df.reset_index().predicted, label="predicted", color=pred_color)

    # Set the legend
    actual_patch = mpatches.Patch(color=actual_color, label="actual values")
    pred_patch = mpatches.Patch(color=pred_color, label="predicted values")
    plt.legend(handles=[actual_patch,pred_patch])
    
    plt.savefig(outfile)


###################
## Classifier plots
###################

def plot_roc(store_fpr, store_tpr, aucs, plt_prefix='', roc_fig_file=None):
    '''
    plot auroc curve(s) with mean and std; save if a roc_fig_file is provided

        INPUTS:
            * store_fpr, store_tpr, aucs: a list where each element represents data for one curve
            * plt_prefix: label to add to the title of plot
            * roc_fig_file: full path to save file
            * savefig: boolean to save or not save figure

        note: first three must be a list; will not plot mean and std if only one curve
    '''
    print("Creating roc plot...")
    plt.close()
    interp_fpr = np.linspace(0, 1, 100)
    store_tpr_interp = []
    ax = plt.figure()
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)

    # plot each cv iteration
    for cv_iter, fpr_tpr_auc in enumerate(zip(store_fpr, store_tpr, aucs)):
        # set_trace()
        fpr, tpr, auc = fpr_tpr_auc
        plt.plot(fpr, tpr, lw=1, alpha=0.5, label="#{}(AUC={:.3f})".format(cv_iter, auc))

        lin_fx = interp1d(fpr, tpr, kind='linear')
        interp_tpr = lin_fx(interp_fpr)

        # store_tpr_interp.append(np.interp(mean_fpr, fpr, tpr))
        # store_tpr_interp[-1][0] = 0.0
        store_tpr_interp.append(interp_tpr)

    # plot mean and std only if more than one curve present
    if len(store_fpr) != 1:
        # plot mean, sd, and shade in between
        mean_tpr = np.mean(store_tpr_interp, axis=0)
        # mean_tpr[-1] = 1.0
        mean_auc = metrics.auc(interp_fpr, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(interp_fpr, mean_tpr, color='b',
                 label="Mean(AUC={:.2f}+/-{:.2f})".format(mean_auc, std_auc),
                 lw=2, alpha=.8)

        std_tpr = np.std(store_tpr_interp, axis=0)
        tprs_upper = np.minimum(mean_tpr + 2*std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - 2*std_tpr, 0)
        plt.fill_between(interp_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                         label="+/- 2 S.D.")

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for {}'.format(plt_prefix))
    plt.legend(loc="lower right")

    if roc_fig_file:
        plt.savefig(roc_fig_file)
        print("Done. AUROC curve saved to: {}".format(roc_fig_file))

    return ax


def plot_pr(precisions, recalls, avg_prs, plt_prefix, pr_fig_file=None, pos_prop=None):
    ''' plot PR curve(s) with mean and std; save if pr_fig_file is provided

        INPUTS:
            * precisions, recalls, avg_prs: must be a list where each element represents data for one curve
            * plt_prefix: label to add to the title of plot
            * pr_fig_file: full path to save file
            * pos_prop: total true positives / total samples (i.e. proportion of positves)

        note: first three must be a list; will not plot mean and std if only one curve
    '''
    plt.close()
    print("Creating PR curve plot ...")
    # mean_rc = np.linspace(0, 1, 100)
    interp_rc = np.linspace(0, 1, 100)

    store_pr_interp = []
    ax = plt.figure()

    # plot line of random chance
    if pos_prop:
        plt.plot([0, 1], [pos_prop, pos_prop], linestyle='--', lw=2,
                 color='r', label='Chance({:.3f})'.format(pos_prop), alpha=.8)

    # plot each cv_iter
    for cv_iter, pr_rc_avg in enumerate(zip(precisions, recalls, avg_prs)):

        pr_array, rc_array, pr_avg = pr_rc_avg
        # plt.plot(rc_array, pr_array, lw=1, color='k', alpha=0.4)
        plt.step(rc_array, pr_array, lw=1, alpha=0.8, where='post', label="#{}(AvgPR={:.3f})".format(cv_iter, pr_avg))

        # interpolate recall to have the same length array for taking mean
        lin_fx = interp1d(rc_array, pr_array, kind='linear')
        interp_pr = lin_fx(interp_rc)
        store_pr_interp.append(interp_pr)

    # set_trace()

    # plot mean and std only if more than one curve present
    if len(precisions) != 1:
        # mean and std
        mean_pr = np.mean(store_pr_interp, axis=0)
        mean_avg_pr = np.mean(avg_prs)
        std_avg_pr = np.std(avg_prs)

        # std of each pr-curve
        std_pr = np.std(store_pr_interp, axis=0)
        pr_upper = np.minimum(mean_pr + 2*std_pr, 1)
        pr_lower = np.maximum(mean_pr - 2*std_pr, 0)
        plt.fill_between(interp_rc, pr_lower, pr_upper, color='grey', alpha=.2,
                         label="+/- 2 S.D.")

        plt.plot(interp_rc, mean_pr, color='b',
                 label="Mean(AUC={:.2f}+/-{:.2f})".format(mean_avg_pr, std_avg_pr), lw=2, alpha=0.8)

    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR Curve for {}'.format(plt_prefix))
    plt.legend(loc="lower right")

    if pr_fig_file:
        plt.savefig(pr_fig_file)
        print("PR curve saved to: {}".format(pr_fig_file))

    return ax

#############
## Misc plots
#############

def barplot_feat_importance(feat_df, top_n, model_id, outfile): 
    plt.close()
    '''
    Selects the top_n feature to plot as barplot of importances to png

    INPUTS: 
        - feat_df: each row is a feature, with following columns = ['feature','improtance','std']
        - top_n: n number of top features to plot 
        - model_id: prefix specifying model info
        - fig_file: full path to save fig
 
    '''
    print("Calc feature importances...")

    # select top_n features after sorting
    top_n = min(top_n,feat_df.shape[0]) # will crash if top_n>total features
    feat_df.rename(columns={'average' : 'importance'},inplace=True)
    top_df = feat_df.sort_values('importance', ascending=False).iloc[0:(top_n)]
    
    ax = plt.figure()
    plt.errorbar(np.arange(0,top_n),top_df.importance.values, yerr=top_df.loc[:,'std'].values, fmt='o', color='black', ecolor='lightgray', elinewidth=3, capsize=0)
    plt.xticks(np.arange(0,top_n), top_df.feature.values, rotation=90)
    plt.xlabel('features')
    plt.ylabel('importance')
    plt.title('Feature Importance: {}\n(top {} features)'.format(model_id, top_n))
    plt.tight_layout()
    plt.savefig(outfile)
    print("Feature importance plot saved to: {}".format(outfile))

    return ax
