import pandas as pd
import numpy as np
import sys
import os

cutoff = 0.99

try:
    infilename = sys.argv[1]
except:
    sys.exit("python3 process_features.py feature_file.tab comma_separated_columnnames_to_ignore [Default= sid,label]")

try:
    ignores = sys.argv[2].strip().split(",")
except:
    ignores = ["sid","label"]

def feats(df):
    return [x for x in list(df) if x not in ignores]
    
assert os.path.isfile(infilename), "{} not found".format(infilename)
outfilename = os.path.splitext(os.path.basename(infilename))[0] + "_proc.tab"
df = pd.read_csv(infilename,sep="\t")
print("Initial df contains {} considered features".format(len(feats(df))))

# lazy check if there's a label column with all the same values
if 'label' in list(df) and len(df.label.unique()) == 0:
    print("Warning: labels column has only one value")

dropcols = list()

# Filter columns with 0 variance
print("Checking for columns with no variance")
dfvar = df.var()
for c in feats(df):
    if dfvar[c]==0: dropcols.append(c)
df = df[[x for x in list(df) if x not in dropcols]]
if len(dropcols)>0:
    print("{} columns with 0 variance dropped".format(len(dropcols)))
    # Write to outfile in case user breaks on next step
    df.to_csv(outfilename,sep="\t",header=True,index=False)

# Filter columns with high correlation (0.95)
# Let the user know if there may be issues
print("Checking for correlated columns (r2>{})".format(cutoff))
if len(feats(df)) > 1000:
    print("Warning, # features exceeds 1000, this may be too large for memory.")
corr_matrix = df[feats(df)].corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(np.bool))
c_to_drop = [column for column in upper.columns if any(upper[column] > cutoff)]
df = df[[x for x in list(df) if x not in c_to_drop]]
if len(c_to_drop)>0:
    print("{} correlated columns dropped".format(len(c_to_drop)))
    df.to_csv(outfilename,sep="\t",header=True,index=False)
elif len(c_to_drop)==0 and len(dropcols)==0:
    sys.exit("No columns dropped, no new file created")
print("Final df contains {} features".format(len(feats(df))))
