# Script to get dataset stats such as node and edge counts from the raw data
import pandas as pd
from os import listdir

attribute_df = pd.DataFrame(columns=['Count','Expected','Actual','Diff'])

# Read in raw files
dfs = {}
for f in listdir(): 
    if f.endswith('.csv') and f.startswith('bio-decagon') and 'targets-all' not in f: # 'bio-decagon-targets-all.csv' is not used
        df = pd.read_csv(f)
        name = f.split('.')[0].split('-')[-1]
        dfs[name] = df

# Drug nodes
drug_IDs = set()
for df in dfs.values():
    for col in df.columns:
        if 'STITCH' in col:
            for ID in df[col].unique():
                drug_IDs.add(ID)
assert all([ID.startswith('CID') for ID in drug_IDs])
drug_expected = 645
attribute_df.loc[len(attribute_df)] = ['Drug nodes', drug_expected, len(drug_IDs), len(drug_IDs)-drug_expected]

# Protein nodes
prot_IDs = set()
for df in dfs.values():
    for col in df.columns:
        if 'Gene' in col:
            for ID in df[col].unique():
                prot_IDs.add(ID)

assert [int(ID) for ID in prot_IDs] # Test they can be converted to int
prot_expected = 19085
attribute_df.loc[len(attribute_df)] = ['Protein nodes', prot_expected, len(prot_IDs), len(prot_IDs)-prot_expected]

# Edge stats. Expected values are those reported in the Decagon paper.
expected_ppi = 715612
attribute_df.loc[len(attribute_df)] = ['PPI edges', expected_ppi, len(dfs['ppi']), len(dfs['ppi'])-expected_ppi]
expected_polySide = 4651131
attribute_df.loc[len(attribute_df)] = ['Polypharmic side effect edges', expected_polySide, len(dfs['combo']), len(dfs['combo'])-expected_polySide]
expected_drugTarget = 18596
attribute_df.loc[len(attribute_df)] = ['Drug-target edges', expected_drugTarget, len(dfs['targets']), len(dfs['targets'])-expected_drugTarget]


##TODO: Get other graph statistics e.g. density, mean path length etc

# Save stats
attribute_df.to_csv('dataset_attributes.csv', index=False)