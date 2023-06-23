import pandas as pd

# Create storage dataframe
out_edges = pd.DataFrame(columns=['head', 'relation', 'tail'])

# Add PPI edges
ppi = pd.read_csv('../raw/bio-decagon-ppi.csv')
ppi['relation'] = 'ProteinProteinInteraction'
ppi.columns = ['head', 'tail', 'relation']
out_edges = ppi[['head', 'relation', 'tail']]
del ppi

# Add drug-target edges
drug_target = pd.read_csv('../raw/bio-decagon-targets.csv')
drug_target['relation'] = 'DrugTarget'
drug_target.columns = ['head', 'tail', 'relation']
out_edges = out_edges.append(drug_target, ignore_index=True)
del drug_target

# Add monopharmic side effect edges
monoSE = pd.read_csv('../raw/bio-decagon-mono.csv').drop(columns=['Side Effect Name'])
monoSE[2] = monoSE.STITCH
monoSE.columns = ['head', 'relation', 'tail']
out_edges = out_edges.append(monoSE, ignore_index=True)
del monoSE

# Add Polypharmic side effect edges
polySE = pd.read_csv('../raw/bio-decagon-combo.csv').drop(columns=['Side Effect Name'])
polySE.columns = ['head', 'tail', 'relation']
out_edges = out_edges.append(polySE, ignore_index=True)
del polySE

# Save edges
out_edges.to_csv('full_edgelist_selfloops.tsv', sep='\t', header=None, index=False)