import pandas as pd
import argparse
import multiprocessing as mp
import networkx as nx
import numpy as np

# Get specified target file
parser = argparse.ArgumentParser()
parser.add_argument('edgelist_tsv')
parser.add_argument('--output_dir')
args = parser.parse_args()


def construct_graph(nodes, edges):
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    return graph


def get_edge_stats(df, rel, rel_nodes_dict):
    edges = df.query('relation == @rel')[['head', 'tail']].to_numpy()
    count = len(edges)
    if rel != 'DrugTarget':
        g = construct_graph(rel_nodes_dict[rel], edges)
        density = nx.density(g)
        transitivity = nx.transitivity(g)
    else:
        g = nx.Graph()
        g.add_nodes_from(rel_nodes_dict[rel][0], bipartite=0)
        g.add_nodes_from(rel_nodes_dict[rel][1], bipartite=1)
        g.add_edges_from(edges)
        num_dyads = len(rel_nodes_dict[rel][0]) * len(rel_nodes_dict[rel][1])
        density = count / num_dyads
        transitivity = None
    n_components = nx.number_connected_components(g)
    try:
        diameter = nx.diameter(g)
    except nx.exception.NetworkXError:
        largest_comp = sorted(
            nx.connected_components(g),
            key=len, reverse=True
        )[0]
        diameter = nx.diameter(g.subgraph(largest_comp))

    return ['edge', rel, count, density, n_components, diameter, transitivity]


# Load data
file_loc = args.edgelist_tsv
edges = pd.read_csv(
    file_loc, header=None, sep='\t', dtype={0: str, 1: str, 2: str}
)
edges.columns = ['head', 'relation', 'tail']

# Get set of all nodes
drug_nodes = set()
protein_nodes = set()
for col in ['head', 'tail']:
    for node in edges[col].unique():
        try:
            protein_nodes.add(str(int(node)))  # Protein nodes are ints
        except ValueError:
            if node.startswith('CID'):
                drug_nodes.add(node)
            else:
                raise ValueError(f'Found node of unknown type: {node}')
n_proteins = len(protein_nodes)
n_drugs = len(drug_nodes)

# Node dict
nodes_per_relation = {}
for rel in edges.relation.unique():
    if rel.startswith('C'):
        nodes_per_relation[rel] = drug_nodes
nodes_per_relation['ProteinProteinInteraction'] = protein_nodes
nodes_per_relation['DrugTarget'] = [drug_nodes, protein_nodes]

# Create output dataframe
cols = [
    'type', 'name', 'count',
    'density', 'num_components',
    'largest_component_diameter',
    'transitivity'
]
node_rows = [
    ['node', 'Total', n_drugs + n_proteins],
    ['node', 'Proteins', n_proteins],
    ['node', 'Drugs', n_drugs]
]
for row in node_rows:
    while len(row) < len(cols):
        row.append(None)
out_df = pd.DataFrame(node_rows, columns=cols)

# Counts of all possible dyads
dyads_counts = {
    'ProteinProteinInteraction': (n_proteins * (n_proteins-1))/2,
    'DrugTarget': n_drugs * n_proteins,
    'SideEffect': (n_drugs * (n_drugs-1))/2
}

# Calculate full graph stats
total_dyads = dyads_counts['ProteinProteinInteraction']
total_dyads += dyads_counts['DrugTarget']
total_dyads += dyads_counts['SideEffect'] * len(edges.relation.unique())
edge_count = len(edges)
total_density = edge_count / total_dyads
full_graph_row = ['edge', 'Total', edge_count, total_density]
while len(full_graph_row) < len(cols):
    full_graph_row.append(None)
out_df.loc[len(out_df)] = full_graph_row

# Add per-relation stats
mp_args = [
    [edges, rel, nodes_per_relation] for rel in edges.relation.unique()
]
with mp.Pool(mp.cpu_count()) as pool:
    edge_stats = pool.starmap(get_edge_stats, mp_args)
out_df = out_df.append(pd.DataFrame(edge_stats, columns=cols))

# Write to disk
edgelist_name = file_loc.split('/')[-1][:-4]
out_name = f'stats_{edgelist_name}.csv'
if args.output_dir:
    out_name = f'{args.output_dir}/{out_name}'
out_df.to_csv(out_name, index=False)
