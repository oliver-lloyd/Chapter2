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

# Edge counts
def edge_count(df, rel):
    print(rel)
    count = len(df.loc[df.relation == rel])
    return [rel, count]

mp_args = [[edges, rel] for rel in edges.relation.unique()]
with mp.Pool(mp.cpu_count()) as pool:
    counts = pool.starmap(edge_count, mp_args)
edge_counts = {result[0]: result[1] for result in counts}

# Get set of all nodes
nodes = set()
for col in ['head', 'tail']:
    for node in edges[col].unique():
        nodes.add(node)

# Get node counts
total = len(nodes)
n_proteins = len([node for node in nodes if not node.startswith('C')]) # Protein IDs are just ints
n_drugs = total - n_proteins

# Create output dataframe
out_df = pd.DataFrame(columns=['type', 'name', 'count'])
out_df.loc[len(out_df)] = ['node', 'Total', total]
out_df.loc[len(out_df)] = ['node', 'Proteins', n_proteins]
out_df.loc[len(out_df)] = ['node', 'Drugs', n_drugs]

out_df.loc[len(out_df)] = ['edge', 'Total', len(edges)]
for count_name in edge_counts:
    out_df.loc[len(out_df)] = ['edge', count_name, edge_counts[count_name]]

# Write to disk
edgelist_name = file_loc.split('/')[-1][:-4]
out_name = f'stats_{edgelist_name}.csv'
if args.output_dir:
    out_name = f'{args.output_dir}/{out_name}'
out_df.to_csv(out_name, index=False)