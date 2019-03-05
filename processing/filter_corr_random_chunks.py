## Filter correlated columns using random chunks of columns
## Will not run into memory/time issues but is not exhaustive and may miss correlations

import pandas as pd
import numpy as np
import sys
import math
import itertools
import os

try:
    chunk = int(sys.argv[2])
except:
    chunk = 100

try:
    cutoff = sys.argv[3]
except:
    cutoff = 0.95    
        
try:    
    infile = sys.argv[1]
except:
    sys.exit("python3 filter_corr.py feature_file.tab [column_chunk_size=100] [corr_cutoff=0.95]")
df = pd.read_csv(infile,sep="\t")
ncol = df.shape[1]
print("Renaming {} to {}.old".format(infile,infile))
os.rename(infile,"{}.old".format(infile))
print("original number of columns = {}".format(ncol))
collist = list(df.columns)
to_drop = list()

for i in range(1,1001):
    subcol = list(np.random.choice(collist,chunk,False))
    current = df[subcol]
    corr_matrix = current.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(np.bool))
    c_to_drop = [column for column in upper.columns if any(upper[column] > cutoff)]
    to_drop.append(c_to_drop)
    print("finished round {} of 1000".format(i))
    if len(c_to_drop)>0:
        print(c_to_drop)
print("Running through final chunks in order")
col_a = 0
col_b = min(ncol-1, col_a + chunk)
while col_b < ncol - 1:
    current = df.iloc[:,col_a:col_b]
    corr_matrix = current.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(np.bool))
    c_to_drop = [column for column in upper.columns if any(upper[column] > cutoff)]
    to_drop.append(c_to_drop)
    print("finished {} to {}".format(col_a,col_b))
    col_a += chunk
    col_b = min(ncol-1, col_a + chunk)
current = df.iloc[:,col_a:col_b]
corr_matrix = current.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(np.bool))
c_to_drop = [column for column in upper.columns if any(upper[column] > cutoff)]
to_drop.append(c_to_drop)      
df.drop(list(set(itertools.chain(*to_drop))),axis=1,inplace=True)
df.to_csv(infile,sep="\t",header=True,index=False)    
print("Final number of columns = {}".format(df.shape[1]))
df.to_csv(infile,sep="\t",header=True,index=False)
