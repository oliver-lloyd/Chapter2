import pandas as pd
import multiprocessing as mp

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
monoSE['relation'] = 'MonopharmacySE'
monoSE.columns = ['head', 'tail', 'relation']
out_edges = out_edges.append(monoSE, ignore_index=True)
del monoSE

# Add Polypharmic side effect edges
polySE = pd.read_csv('../raw/bio-decagon-combo.csv', nrows=10000).drop(columns=['Side Effect Name'])

def create_multidrug(df):
    df['head'] = [f'{row["STITCH 1"]}-{row["STITCH 2"]}' for i, row in df.iterrows()]
    return df

n_cpu = mp.cpu_count()
n_rows = polySE.shape[0]
chunk_size = int(n_rows/n_cpu)
chunks = [polySE.iloc[polySE.index[i:i + chunk_size]] for i in range(0, n_rows, chunk_size)]
del polySE
with mp.Pool(n_cpu) as pool:
    new_chunks = pool.map(create_multidrug, chunks)
del chunks

multidrug_set = set()
for chunk in new_chunks:
    chunk = chunk.drop(columns=['STITCH 1', 'STITCH 2'])
    chunk['relation'] = 'PolypharmacySE'
    chunk.columns = ['tail', 'head', 'relation']
    for multidrug in chunk['head'].unique():
        multidrug_set.add(multidrug)
    out_edges = out_edges.append(chunk, ignore_index=True)
del new_chunks

# Create binary edges for the single drugs that make up a multidrug
def multidrug_comprises(multidrug):
    out = []
    for drug in multidrug.split('-'):
        out.append([multidrug, 'MultidrugContains', drug])
    return out 

with mp.Pool(n_cpu) as pool:
    comprise_rows = pool.map(multidrug_comprises, multidrug_set)
flat_rows = [row for sublist in comprise_rows for row in sublist]
comprise_df = pd.DataFrame(flat_rows, columns=['head', 'relation', 'tail'])
out_edges = out_edges.append(comprise_df, ignore_index=True)
del comprise_df

# Save edges
out_edges.to_csv('full_edgelist_multidrugs.tsv', sep='\t', header=None, index=False)
