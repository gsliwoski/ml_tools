## The longest but least memory intensive
## Checks every column pair one at a time

import pandas as pd
import numpy as np
import sys
import math
import itertools
import os

try:
    cutoff = sys.argv[2]
except:
    cutoff = 0.95  

try:    
    infile = sys.argv[1]
except:
    sys.exit("python3 filter_corr_iterative.py feature_file.tab [corr_cutoff=0.95]")

df = pd.read_csv(infile,sep="\t")
ncol = df.shape[1]
print("Renaming {} to {}.old".format(infile,infile))
os.rename(infile,"{}.old".format(infile))
print("original number of columns = {}".format(ncol))
colset = set(df)
colset.remove('sid')
#print(colset)
remove = set()
n = 0
colset2 = colset.copy()
for x in colset:
    n += 1        
    if n%1001==0:
        print("{} complete".format(n-1))
    if x in remove: continue
    colset2.remove(x)
    colset2 = colset2 - remove    
    z = 0
    for y in colset2:
        z += 1
        try:
            if np.corrcoef(df[x],df[y])[0][1]>cutoff:
                remove.add(y)
                print(y)
        except ValueError:
            continue                

df.drop(list(remove),axis=1,inplace=True)
print("final column count = {}".format(df.shape[1]))
df.to_csv(infile,sep="\t",header=True,index=False)
            
