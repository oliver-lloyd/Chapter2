import pandas as pd
import multiprocessing as mp

# Create storage dataframe
out_edges = pd.DataFrame(columns=['head', 'relation', 'tail'])

# Load core network
core = pd.read_csv('../../processed/core_network_ppi_drugtarget.tsv', header=None, sep='\t', dtype={0:str, 1:str, 2:str})
core.columns = ['head', 'relation', 'tail']
out_edges = out_edges.append(core)
del core

# Add monopharmic side effect edges
monoSE = pd.read_csv('../../processed/monopharmacy_edges.tsv', header=None, sep='\t')
monoSE.columns = ['head', 'relation', 'tail']
out_edges = out_edges.append(monoSE, ignore_index=True)
del monoSE

# Add Polypharmic side effect edges
polySE = pd.read_csv('../../processed/polypharmacy_edges.tsv', header=None, sep='\t')
polySE.columns = ['head', 'relation', 'tail']

def create_multidrug(df):
    df['head'] = [f'{row["head"]}-{row["tail"]}' for i, row in df.iterrows()]
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
    chunk['tail'] = chunk['relation']
    chunk['relation'] = 'PolypharmacySideEffect'
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
