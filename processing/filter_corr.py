import pandas as pd
import numpy as np
import sys
import math
import itertools
import os

try:
    nrounds = sys.argv[2]
except:
    nrounds = 5

try:
    cutoff = sys.argv[3]
except:
    cutoff = 0.95    
        
try:    
    infile = sys.argv[1]
except:
    sys.exit("python3 filter_corr.py feature_file.tab [n_steps=5] [corr_cutoff=0.95]")
df = pd.read_csv(infile,sep="\t")
ncol = df.shape[1]
print("Renaming {} to {}.old".format(infile,infile))
os.rename(infile,"{}.old".format(infile))
print("original number of columns = {}".format(ncol))
for step in range(1,nrounds+1)[::-1]:
    ncol = df.shape[1]
    print("outer step {}".format(step))
    step_size = int(ncol/step)
    if step_size>15000: continue
    to_drop = list()
    for inner_step in range(step):
        col_a = inner_step * step_size
        col_b = col_a + step_size
        print("{} to {}".format(col_a,col_b))
        current = df.iloc[:,col_a:col_b]
        corr_matrix = current.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(np.bool))
        c_to_drop = [column for column in upper.columns if any(upper[column] > cutoff)]
        print("completed {} to {}".format(col_a,col_b))
        to_drop.append(c_to_drop)
    df.drop(list(itertools.chain(*to_drop)),axis=1,inplace=True)
    df.to_csv(infile,sep="\t",header=True,index=False)    
print("Final number of columns = {}".format(df.shape[1]))
df.to_csv(infile,sep="\t",header=True,index=False)

