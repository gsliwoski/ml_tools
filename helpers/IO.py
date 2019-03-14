import os
import argparse
import math
import sys
import time
import pandas as pd
from datetime import datetime
sys.path.append('/dors/capra_lab/users/sliwosgr/cancer_project/random_forest/')
import estimator_plots

##############
## DEFAULTS ##
##############

# Default parameters for classifier
default_class_parameters = {
                            'rf':
                                {
                                'max_depth': 10,
                                'class_weight': 'balanced',
                                'n_estimators': 10
                                },
                            'ann':
                                  {
                                  'hidden_layer_sizes': (50,),
                                  'solver': 'lbfgs'
                                  }
                            }
                            
# Default parameters for Regressor
default_reg_parameters = {
                          'rf':
                              {
                              'max_depth': 10,
                              'n_estimators': 10
                              },
                          'ann':
                               {
                               'hidden_layer_sizes': (50,),
                               'solver': 'lbfgs'                             
                               }
                          }

# Default parameters for cross validation
default_cv_parameters = {
                         'n_splits': 10,
                         'train_size': 0.8,
                         'test_size': 0.2
                         }
###########
## INPUT ##
###########

ROOTDIR = "/dors/capra_lab/users/sliwosgr/ml_tools/"
DATE = datetime.now().strftime('%Y-%m-%d')

# learner = 'ann' or 'rf' for random forest
def initialize(learner):
    parser = argparse.ArgumentParser(description='train classifier with params provided and report performance.')
    parser.add_argument('--sampleid', '-s', action='store', type=str, default='sid',
                        help='name of the id column that is unique for each row and is used to join features and labels. Default: sid')
    parser.add_argument('--features','-f', action='store', type=str, required=True,
                        help='tsv file containing features includes sample id column and 1 or more feature columns')
    parser.add_argument('--labels', '-l', action='store', type=str, required=True,
                        help='tsv file containing labels includes sample id column and 1 or more label columns')
    parser.add_argument('--label_column', '-c', action='store', type=str, default='label',
                        help='column header for label of interest found in only label file. Default: label')
    parser.add_argument('--params','-p', action='store', type=str, default=None,
                        help='file with custom parameters, one per line, param_name: param_value. If not provided uses defaults')
    parser.add_argument('--grid','-g', action='store', type=str, default=None,
                        help='paramer file that includes multiple values per parameter which launches hyperparameter optimization')
    parser.add_argument('--output','-o', action='store', type=str, default=None,
                        help='output suffix. Default is paramfilename_labelfilename_classifiertype')
    parser.add_argument('--outpath', '-i', action='store', type=str, default='output/',
                        help='path to write output files to. Default: output/')
    parser.add_argument('--regressor', '-r', action='store_true', help='indicate classifier is regressor rather than classifier')
    parser.add_argument('--date', '-d', action='store_true', help='attach date to output suffix')
    parser.add_argument('--random_state', '-x', action='store', type=int, default=32,
                        help='set a random seed, if below 0, does not set a random seed. Does not override if set in params file. Default: 32')
    

    # Initialization checks
    results = parser.parse_args()
    assert not (results.params is not None and results.grid is not None), "must provide either grid or params file, not both"
    if results.params is not None:
        paramfile = results.params
        noopt = True
    elif results.grid is not None:
        paramfile = results.grid
        noopt = False
    else:
        print("Using default parameters")
        noopt = True
        paramfile = 'defaultreg' if results.regressor else 'defaultclass'    
    if results.params is not None:
        assert os.path.isfile(paramfile), "{} not found".format(paramfile)
    assert os.path.isfile(results.features), "{} not found".format(results.features)
    assert os.path.isfile(results.labels), "{} not found".format(results.labels)

    # Load parameters
    loaded_parameters = load_parameters(paramfile, noopt)

    # Add any missing default parameters:
    if results.regressor:   
        parameters = default_reg_parameters[learner]
    else:
        parameters = default_class_parameters[learner]
    parameters.update(default_cv_parameters)
    parameters.update(loaded_parameters)        

    # Add the random state if there is one

    if results.random_state > 0 and 'random_state' not in parameters:
        parameters['random_state'] = results.random_state
    
    # Define the output suffix
    model_type = 'reg' if results.regressor else 'class'
    if results.params is None:
        p = 'default'
    else:
        p = os.path.splitext(os.path.basename(results.params))[0]
    l = os.path.splitext(os.path.basename(results.labels))[0]
    f = os.path.splitext(os.path.basename(results.features))[0]
    if results.output is None:
        output_suffix = "{}_{}_{}_{}".format(p,f,l,model_type)
    else:
        output_suffix = results.output
    if results.date:
        output_suffix += "_{}".format(DATE)
    if output_suffix=="":
        oc = ""
    else:
        oc = "-"

    # Define output path
    if results.outpath == 'output/':
        output_dir = os.path.join(os.getcwd(),results.outpath)
    else:
        output_dir = results.outpath
    if not os.path.isdir(output_dir):
        try:
            os.mkdir(output_dir)
        except: # In case running parallel
            assert os.path.isdir(output_dir), "Failed to create {}".format(output_dir)

    # Define output files
    outfiles = {
                'ROC_FIG_FILE': "roc{}{}{}.png",
                'PR_FIG_FILE': "pr{}{}{}.png",
                'FEAT_IMPORT_FILE': "feature_importance{}{}{}.tab",
                'FEATURE_FIG_FILE': "feature_importance_plot{}{}{}.png",
                'PREDICTIONS_FILE': "predictions{}{}{}.tab",
                'TRAIN_METRIC_DF_FILE': "train_metrics{}{}{}.tab",
                'TEST_METRIC_DF_FILE': "test_metrics{}{}{}.tab",
                'OUTPUT_DIR': output_dir
                }
    for x in outfiles:
        outfiles[x] = os.path.join(output_dir,outfiles[x].format("_{}".format(learner), oc, output_suffix))
    outfiles["PERFORMANCE_FILE"] = os.path.join(output_dir,"{}_{}_performance.log".format(learner,model_type))

    # Load the labels and features
    labels_df = load_labels(results)
    X_mat, y_labels, _ = load_X_y(results, labels_df)  

    # Define a model ID for plots
    model_id = "Random Forest{}{}".format(oc,output_suffix)
    
    # Return initialized datasets and paths
    return X_mat, y_labels, parameters, model_type, noopt, outfiles, model_id

def load_labels(results):
    """
    Load the labels file and filter for the desired label
    
    Params
    ------
    results : argparser object

    Returns
    -------
    final_labels_df: pandas.DataFrame
        Dataframe with 2 columns: sid, label
    """

    # load labels
    print("Loading labels...")
    try:    
        labels_df = pd.read_csv(results.labels, sep="\t")
    except:
        sys.exit("Failed to parse labels file {} with pandas".format(results.labels))
    assert labels_df.shape[0] > 0 and labels_df.shape[1] > 0, "labels df is empty"                
    assert results.label_column in list(labels_df), "column {} not found in {}".format(results.label_column,
                                                                                               results.labels)
    assert results.sampleid in list(labels_df), "column {} not found in {}".format(results.sampleid,
                                                                                           results.labels)

    # Normalize the column names
    labels_df = labels_df.rename(columns = {results.label_column: 'label', results.sampleid: 'sid'})[['sid','label']]

    # Print label information depending on type of model
    mp = "Regressor" if results.regressor else "Classifier"
    print("{} labels summary:".format(mp))
    if results.regressor:
        print(labels_df.label.describe())
    else:        
        for x in labels_df.groupby('label'):
            l,r = x
            print("{}: {}".format(l,r.shape[0]))
        print("total: {}".format(labels_df.shape[0]))
    
    return labels_df

def load_X_y(results, label_df):
    """
        Load features and then output 'X_mat', 'y_labels' and full 'merged_df'.

        Params
        ------
        results : argparse object
        label_df : pandas.DataFrame
            one row per 'sid' and its desired label

        Returns
        -------
        X_mat : numpy.array
            sample by FEATURE array with values from feature file
        y_labels : numpy.array (sample x 1 )
            labels of interest
        merged_df : pandas.DataFrame
            full dataframe used to create X_mat and y_labels with headers
    """
#    stime = time.time()
    print("Loading X and y matrices...")

    # load feature matrix
    try:
        feat_df = pd.read_csv(results.features, sep="\t")
    except:
        sys.exit("Failed to parse feature file {} with pandas".format(results.features))
    print(list(feat_df))
    assert results.sampleid in list(feat_df), "column {} not found in feature file {}".format(results.sampleid, results.features)
    assert feat_df.shape[0] > 0, "features df is empty"
    assert feat_df.shape[1] > 1, "features file contains no features"

    # Handle the possibility that features dataframe contains labels or
    # a feature has the name 'label'
    if results.label_column in list(feat_df):
        print("Dropping label column {} from features_df".format(results.label_column))
        feat_df.drop(results.label_column, inplace=True, axis=1)
    if 'label' in list(feat_df): # If there is already a !!feat_label!! column then oh well crasssh
        print("Warning, renaming the feature 'label' to '!!feat_label!!' to avoid merge clashes")
        feat_df.rename(columns={'label':'!!feat_label!!'}, inplace=True)
    
    # Normalize the sample identifier
    feat_df.rename(columns={results.sampleid : 'sid'}, inplace=True)
    
    print("loaded {} features".format(feat_df.shape[1]-1))
    
    # merge label to feat_df
    merged_df = pd.merge(feat_df, label_df, how='inner', on='sid')
    merged_df.drop_duplicates(inplace=True)

    # Check for any inconsistencies after merge
    flost = feat_df.shape[0] - merged_df.shape[0]
    llost = label_df.shape[0] - merged_df.shape[0]
    if flost > 0:
        print("Warning: merged df has {} rows less than features df, some samples lost".format(flost))
    if llost > 0:
        print("Warning: merged df has {} rows less than labels df, some samples lost".format(llost))
    if flost < 0 or llost < 0:
        print("WARNING: there are more rows in merged df than labels df or features df, incompatible duplication during merge!")
    
    # Separate features and labels and return
    X_mat = merged_df.drop(['sid','label'],axis=1, inplace=False)
    y_labels = merged_df.label
    print("Done creating X_mat {} and y_labels.".format(X_mat.shape))
    return X_mat, y_labels, merged_df

def load_parameters(paramfile, noopt):
    """
        Load a parameter file if provided.
        If no file provided, load defaults based on whether it's .
        Parse file as single set of params or grid if optimization is provided
        
        Params
        ------
        full path to parameter file (or None for default)
        noopt: Bool specifying whether it's a grid file or single set of parameters
        
        Returns
        -------
        parameters : Dict
            keys are param labels and values are either single values or lists for optimization
    """
                         
    if paramfile == 'defaultreg':
        parameters = default_reg_parameters
        p = 'defaults'
    elif paramfile == 'defaultclass':
        parameters = default_class_parameters
        p = 'defaults'
    else:
        infile = open(paramfile)
        if noopt:
            parameters = parse_paramfile(infile)
        else:
            parameters = parse_gridfile(infile)
        infile.close()
        if noopt:
            p = "file"
        else:
            p = "grid"
    print("Read parameter {}".format(p))
    for x,y in parameters.items():
        print(f"{x}: {y}")             
    return parameters       
                  
def parse_gridfile(infile):
    """
        Load a parameter grid definition file:
            Each line is a parameter argument to use in grid and colon and values (eg: max_depth: 10,20)
            Values can be csv or a range defined by (min - max, step_size) [will always include min and max though]
            Example of range = max_depth: 1 - 10, 2 (will create 1,3,5,7,9,10)
            Any repeated keys will be skipped
            
        Params
        ------
        ostream to param gridfile
        
        Returns
        -------
        param_grid : Dict
            keys are param labels and values are lists            
    """
    param_grid = dict()
    for line in infile:
        line = line.strip().split("#")[0]
        line = line.split(":")
        if len(line)<2: continue
        label = line[0]
        if label in param_grid: continue
        vals = line[1]
        if "-" in vals:
            try:                
                vals = vals.split("-")
                try:
                    a = int(vals[0])
                except ValueError:
                    a = float(vals[0])
                try:
                    b = int(vals[1].split(",")[0])
                except:
                    b = float(vals[1].split(",")[0])                        
                try:                        
                    step_size = vals[1].split(",")[1]
                    try:
                        step_size = int(step_size)
                    except:
                        step_size = float(step_size)
                except IndexError:
                    step_size = 1
                vals = [a]
                while (vals[-1] + step_size) < b:
                    vals.append(vals[-1]+step_size)
                vals.append(b)                                                                          
            except ValueError:
                pass          
        else:
            tmpv = list()
            for x in vals.split(","):
                try:
                    tmpv.append(int(x))
                except ValueError:
                    try:
                        tmpv.append(float(x))
                    except ValueError:
                        x = x.strip()
                        if x == 'True':
                            x = True
                        elif x == 'False':
                            x = False                            
                        tmpv.append(x)
            vals = tmpv
        param_grid[label] = vals
    return param_grid

def parse_paramfile(infile):
    '''
    Loads single parameters used for training model with no optimization
    Each line is a parameter flag: value (eg: max_depth: 20)        
    Only the last entry for a parameter is kept if there are repeats

        Params
        ------
        ostream to parameter file
        
        Returns
        -------
        param_grid : Dict
            items are param label keys and values
    '''
    params = dict()
    for line in infile:
        line = line.split("#")[0]
        line = [x.strip() for x in line.split(":")]
        if len(line)!=2: continue
        try:
            val = int(line[1])
        except ValueError:
            try:
                val = float(line[1])
            except ValueError:
                val = line[1]
        params[line[0]] = val
    return params                                                                    

############
## OUTPUT ##
############
    

def record_performance(metric_df, model_id, outfile, reg):
    if reg:
        cols = ['r2','explained_variance','mae','mse']
    else:
        cols = ['']  
    cols = metric_df.columns
    metric_df['model_id'] = model_id
    cols = ['model_id'] + list(cols)
    if os.path.isfile(outfiles['PERFORMANCE_FILE']):
        metric_df.to_csv(outfiles['PERFORMANCE_FILE'],sep="\t",columns=cols,index=False,header=False,mode='a')
    else:
        metric_df.to_csv(outfiles['PERFORMANCE_FILE'],sep="\t",columns=cols,index=False)

def write_test_predictions(predictions, outfile):
    for i,x in enumerate(predictions):
        x.rename(columns={'true' : 'cv_{}_true'.format(i),
                          'predicted' : 'cv_{}_predicted'.format(i)}, inplace=True)
    allpred = pd.concat(predictions, axis=1, ignore_index=True)
    allpred.to_csv(outfile, header=True, index=False, sep="\t")

    # plot curves
#    pos_prop = np.sum(y_test == 1)/len(y_test)
#    plot_roc([metrics_results['fpr']], [metrics_results['tpr']], [metrics_results['roc_auc']],
#             output_suffix, roc_fig_file=ROC_FIG_FILE)
#    plot_pr([metrics_results['pr_curve']], [metrics_results['rc_curve']], [metrics_results['avg_prec']],
#            output_suffix, pr_fig_file=PR_FIG_FILE, pos_prop=pos_prop)
