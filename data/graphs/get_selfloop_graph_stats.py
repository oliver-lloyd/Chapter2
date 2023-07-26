import pandas as pd
import argparse
import multiprocessing as mp

# Get specified target file
parser = argparse.ArgumentParser()
parser.add_argument('edgelist_tsv')
parser.add_argument('--output_dir')
args = parser.parse_args()

# Load data
file_loc = args.edgelist_tsv
edges = pd.read_csv(file_loc, header=None, sep='\t', dtype={0:str, 1:str, 2:str})
edges.columns=['head', 'relation', 'tail']

# Get set of all nodes
nodes = set()
for col in ['head', 'tail']:
    for node in edges[col].unique():
        nodes.add(node)

# Get node counts
total = len(nodes)
n_proteins = len([node for node in nodes if not node.startswith('C')]) # Protein IDs are just ints
n_drugs = total - n_proteins

# Edge counts
def edge_count(df, rel):
    count = len(df.loc[df.relation == rel])
    return count

def get_density(count, rel, num_drugs, num_proteins):
    if rel == 'ProteinProteinInteraction':
        n_dyads = (num_proteins * (num_proteins-1))/2
    elif rel == 'DrugTarget':
        n_dyads = num_drugs * num_proteins
    elif rel.startswith('C'):  # Side effect edges always start with C
        n_dyads = (num_drugs * (num_drugs-1))/2
    else:
        raise ValueError(f'Unrecognised edge type: {rel}')
    
    return count/n_dyads

def get_edge_stats(df, rel, num_drugs, num_proteins):
    count = edge_count(df, rel)
    density = get_density(count, rel, num_drugs, num_proteins)
    return ['edge', rel, count, density]

    


mp_args = [[edges, rel, n_drugs, n_proteins] for rel in edges.relation.unique()]
with mp.Pool(mp.cpu_count()) as pool:
    edge_stats = pool.starmap(get_edge_stats, mp_args)

# Create output dataframe
cols = ['type', 'name', 'count', 'density']
out_df = pd.DataFrame(columns=cols)
out_df.loc[len(out_df)] = ['node', 'Total', total, None]
out_df.loc[len(out_df)] = ['node', 'Proteins', n_proteins, None]
out_df.loc[len(out_df)] = ['node', 'Drugs', n_drugs, None]

edge_count = len(edges)
total_density = edge_count / ( (n_proteins * (n_proteins-1))/2 + (n_drugs * n_proteins) + (n_drugs * (n_drugs-1))/2 )
out_df.loc[len(out_df)] = ['edge', 'Total', edge_count, total_density]
out_df = out_df.append(pd.DataFrame(edge_stats, columns=cols))

# Write to disk
edgelist_name = file_loc.split('/')[-1][:-4]
out_name = f'stats_{edgelist_name}.csv'
if args.output_dir:
    out_name = f'{args.output_dir}/{out_name}'
out_df.to_csv(out_name, index=False)