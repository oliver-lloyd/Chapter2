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
edges = pd.read_csv(
    file_loc, header=None, sep='\t', dtype={0: str, 1: str, 2: str}
)
edges.columns = ['head', 'relation', 'tail']

# Get set of all nodes
nodes = set()
for col in ['head', 'tail']:
    for node in edges[col].unique():
        nodes.add(node)

# Get node counts
total = len(nodes)
n_proteins = len(
    [node for node in nodes if not node.startswith('C')]
)  # Protein IDs are just ints
n_drugs = total - n_proteins

# Counts of all possible dyads
dyads_counts = {
    'ProteinProteinInteraction': (n_proteins * (n_proteins-1))/2,
    'DrugTarget': n_drugs * n_proteins,
    'SideEffect': (n_drugs * (n_drugs-1))/2
}


def edge_count(df, rel):
    count = len(df.loc[df.relation == rel])
    return count


def get_density(count, rel, dyad_count_dict):
    if rel.startswith('C'):
        # All side effect nodes start with the character 'C'
        n_dyads = dyad_count_dict['SideEffect']
    else:
        n_dyads = dyad_count_dict[rel]
    return count/n_dyads


def get_edge_stats(df, rel, dyad_dict):
    count = edge_count(df, rel)
    density = get_density(count, rel, dyad_dict)
    return ['edge', rel, count, density]


mp_args = [
    [edges, rel, dyads_counts] for rel in edges.relation.unique()
]
with mp.Pool(mp.cpu_count()) as pool:
    edge_stats = pool.starmap(get_edge_stats, mp_args)

# Create output dataframe
cols = ['type', 'name', 'count', 'density']
out_df = pd.DataFrame(columns=cols)

# Store node stats
out_df.loc[len(out_df)] = ['node', 'Total', total, None]
out_df.loc[len(out_df)] = ['node', 'Proteins', n_proteins, None]
out_df.loc[len(out_df)] = ['node', 'Drugs', n_drugs, None]

# Calculate full graph stats
total_dyads = dyads_counts['ProteinProteinInteraction']
total_dyads += dyads_counts['DrugTarget']
total_dyads += dyads_counts['SideEffect'] * len(edges.relation.unique())
edge_count = len(edges)
total_density = edge_count / total_dyads
out_df.loc[len(out_df)] = ['edge', 'Total', edge_count, total_density]

# Add per-relation stats
out_df = out_df.append(pd.DataFrame(edge_stats, columns=cols))

# Write to disk
edgelist_name = file_loc.split('/')[-1][:-4]
out_name = f'stats_{edgelist_name}.csv'
if args.output_dir:
    out_name = f'{args.output_dir}/{out_name}'
out_df.to_csv(out_name, index=False)
