# General includes
import sys
import os
import pandas as pd

# PDB includes
from Bio import PDB
from AA import AA,MODRES,AA_PROPERTIES,BB

# Calculation includes
from numpy.linalg import norm as norm
from numpy import mean, vectorize, array
from scipy import spatial
# Note this does not currently support using variant with insertion code
# It also ignores any variants with rescodes when counting variant neighbors
# It only counts each unique position+label once, so if there are 2 clinvar vars
#    at position 100, and 99 is a var, it will only count 1 clinvar at 100
#    but if there is a clinvar and gnomad at 100, 99 will get 1 clinvar and 1 gnomad

# Define distance bins for the various counts
# Include upper bin to get total in structure
BINS = [5,10,15,20,25,30,'total']

# Append a set of normalized features
NORMAL = True

# Ignore hydrogens
HYDROGEN = False

# List of ligand IDs that are potentially artifacts
# For use if ligand counts are added
ARTIFACTS = ["HOH","EDO","EPE","FLC","GAL","GLC","MAN","NH2","PE4","PEG","PG4","MPD",
             "FUC","NDG","NI","PE8","PGE","XY1","GOL","PO4","DOD","IOD","DMS","FMT",
             "BME","ACY","IMD","1PE","NO3","UNL","UNX","LDA","MRD","EOH","IUM","CO3",
             "P6G","GAI","CAC","SCN","DIO","AZI","BCT","HED","URE","CAD","2PE","NH4",
             "H2S","MOH","DMF","PGO","LMT","MPO","SGM","12P","BTB","HTO","UPL","BCN",
             "CDL","HEZ","NET","ARS","CPS","CYN","OLC","BU3","MBO","C8E","MPG","NHE",
             "P33","DMU","SO3","TAM","1PG","LFA","B3P","15P","PO3","TRD","2CV","BNG",
             "YT3","BU1","CCN","CM5","ETE","D12","HEX","PGW","PEO","XPE","LMU","D1D",
             "VO4","144","CXS","TG1","TRT","C10","DPO","HGB","ME2","P4C","PGR","2NV",
             "BF2","BO3","PDO","POL","MSE"]
     
try:
    pdbfile = sys.argv[1]
    chain = sys.argv[2]
    variantfile = sys.argv[3]

except:
    sys.exit("python3 gen_neighbor_features.py pdbfile chainID variant_list.ls (REF(1letter)#ALT(1letter) label per line)")

# Set up IO
assert os.path.isfile(pdbfile), "{} not found".format(pdbfile)
assert os.path.isfile(variantfile), "{} not found".format(variantfile)

pdbroot = os.path.splitext(os.path.basename(pdbfile))[0]
variantroot = os.path.splitext(os.path.basename(variantfile))[0]
outfilename = "{}_{}_neighborhood.tab".format(pdbroot,variantroot)
failoutfile = "{}_{}_neighborhood.failures".format(pdbroot,variantroot)

# Set up variants
variants = dict()
with open(variantfile) as infile:
    for line in infile.readlines():
        if line[0]=="#": continue
        line = line.strip().split()
        if len(line)!=2: continue
        res,label = line
        ref = res[0]
        alt = res[-1]
        try:
            pos = int(res[1:-1])
        except ValueError:
            continue
        if pos not in variants:
            variants[pos] = set()            
        variants[pos].add((ref,alt,label))

if len(variants)==0:
    sys.exit("No usable variants in {}".format(variantfile))

# Set up structure
parser = PDB.PDBParser(PERMISSIVE=1,QUIET=True)
structure = parser.get_structure(pdbroot,pdbfile)
assert chain in structure[0], "chain {} not found in structure".format(chain)
protein = structure[0][chain]
assert len(protein)>0, "no residues found in chain {}".format(chain)

failures = list()
labels = set()

# Process variants
variants_processed = dict()
for position in variants:
    pdbpos = (' ', position, ' ')
    if not protein.has_id(pdbpos):
        for x in variants[position]:
            failures.append("{}{}{} {}".format(x[0],position,x[1],x[2]))
        continue
    try:
        pdbres = AA[protein[position].get_resname()]
    except KeyError:
        pdbres = "?"                            
    variants_processed[pdbpos] = list()
    for ref,alt,label in variants[position]:
        v = "{}{}{}".format(ref,position,alt)
        if ref != pdbres:
            print("Warning, variant {} has ref that doesn't match pdb residue {}".format(v,pdbres))
        variants_processed[pdbpos].append((v,label))
        labels.add(label)     

# Finish setup
if len(failures)>0:
    print("{} variants were not found in chain {}".format(len(failures),chain))                   
    with open(failoutfile,"w") as outfile:
        outfile.write("\n".join(failures)+"\n")
assert len(variants_processed)>0, "No usable variants in structure"
print("Loaded pdbfile and {} variants".format(sum([len(variants_processed[x]) for x in variants_processed])))

# Descriptor functions
def distance(a,b):
    return norm(a-b)

# Calculate the geometric centroid
def centroid(residue):
    # Glycine centroid is CA
    if residue.get_resname()=="GLY":
        return residue['CA'].get_coord()        
    atoms = [a.get_coord() for a in residue \
             if (a.element!="H" or HYDROGEN) \
             and a.get_id() not in BB]
    # If there are no sidechain atoms, use any atoms available
    # Otherwise return None
    if len(atoms)==0:
        atoms = [a.get_coord() for a in residue]
    if len(atoms)==0:
        return None
    else:
        return mean(atoms,axis=0)
            
def get_neighbors(residue, rescen, centroids):
    # Calculate distance to all residues
    distances = spatial.distance.cdist([x[1] for x in centroids], [rescen])

    current_neighbors = {"{}_{}".format(x,y): 0 for x in list(AA_PROPERTIES.keys())+list(labels) for y in BINS}
    for bin in BINS:
        for i,d in enumerate(distances):
            if centroids[i][0].get_id()==residue: continue
            if bin=="total" or d<=bin:
                for label in centroids[i][2]:
                    feat = "{}_{}".format(label,bin)
                    current_neighbors[feat]+=1                    
#    print(distances)
#    print(current_neighbors)
    return current_neighbors

# Get feature based on AA property change at mutation
def mutation_features(vid):
    ref = vid[0]
    alt = vid[-1]
#    mfeats = {"delta_{}".format(x):0 for x in AA_PROPERTIES.keys() if x!='canonical'}
    mfeats = dict()
    for prop in sorted(AA_PROPERTIES.keys()):
        if prop=='canonical': continue
        propcol = "delta_{}".format(prop)
        rp = 1 if ref in AA_PROPERTIES[prop] else 0
        ap = 1 if alt in AA_PROPERTIES[prop] else 0
        mfeats[propcol] = ap-rp
#        mfeats.append(ap-rp)
    return mfeats
        
# Generate dictionary of AA properties
prop_lookup = {x:list() for x in AA.keys() if len(x)==1}
for prop in AA_PROPERTIES.keys():
    for aa in AA_PROPERTIES[prop]:
        prop_lookup[aa].append(prop)
        
# Generate a table of centroids
centroids = list()
for residue in protein:
    if residue.get_id()[0]!=" ": continue
    aa = AA[residue.get_resname()]
    # Generate row of residue id, centroid position, and list of property categories
    current_centroid = [residue,centroid(residue),prop_lookup[aa][:]]
    # If it's also at the position of one or more supplied variants, attach the labels
    failed = current_centroid[1] is None
    if residue.get_id() in variants_processed:
        if failed:
            print("WARNING: variant position {} has no atoms".format(residue.get_id()[1]))
        current_labels = [x[1] for x in variants_processed[residue.get_id()]]
        current_centroid[2]+=current_labels        
    if not failed:
        centroids.append(current_centroid)
#for x in variants_processed:
#    print(x)
#for x in centroids:
#    print(x)

# Generate the header 
header = ['variant','label']
featcols = ["{}_{}".format(x,y) for x in sorted(list(AA_PROPERTIES.keys()))+sorted(list(labels)) for y in BINS] 
mutcols = ["delta_{}".format(x) for x in sorted(AA_PROPERTIES.keys()) if x!='canonical']
header += featcols
  
normheader = ["{}_norm".format(x) for x in featcols if x[-6:]!="_total"]
if NORMAL:
    header += normheader

header += mutcols
    
# Generate the table of variant neighborhood features
#final_features = [header]
final_features = {x:list() for x in header}

for pos in sorted(variants_processed,key=lambda x: x[1]):
    poscen = centroid(protein[pos])
    features = get_neighbors(pos,poscen,centroids)
#    feats = [features[x] for x in featcols]
    # If normalized is set, calculate and add the normalized versions
#    for x in features:
#       final_features[x].append(features[x])
    # Add normalized features
    if NORMAL:
        normfeats = dict()
        for f in features.keys():
            fs = f.rsplit("_",1)
            if fs[1]=='total':continue
            try:
                normf = float(features[f])/features["{}_total".format(fs[0])]
            except ZeroDivisionError:
                normf = 0.0
            normfeats["{}_norm".format(f)] = round(normf,4)
#            final_features["{}_norm".format(f)].append(round(normf,4))
#        feats += [normfeats[x] for x in normheader]
    for v in variants_processed[pos]:
        vid,label = v
        final_features['variant'].append(vid)
        final_features['label'].append(label)
        # Generate a change-specific feature
#        full_feats = feats[:] + mutation_features(vid)
        mf = mutation_features(vid)
        for x in features:
            final_features[x].append(features[x])
        for x in mf.keys():
            final_features[x].append(mf[x])                
        for x in normfeats.keys():
            final_features[x].append(normfeats[x])
#        final_features.append([vid,label]+full_feats)

nfeat = len(featcols) + len(mutcols)
if NORMAL:
    nfeat += len(normheader)

#with open(outfilename,'w') as outfile:
#    for v in final_features:
#        outfile.write(" ".join(str(x) for x in v))
#        outfile.write("\n")

pd.DataFrame.from_dict(final_features).to_csv(outfilename,sep="\t",header=True,index=False,columns=header)

print("wrote {} spatial features to {}".format(nfeat,outfilename))
