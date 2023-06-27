import pandas as pd

file_name = 'full_edgelist_multidrugs.tsv'
edges = pd.read_csv(file_name, header=None, sep='\t', dtype={0:str, 1:str, 2:str})
edges.columns=['head', 'relation', 'tail']

# Edge counts
edge_counts = {}
for rel in edges.relation.unique():
    edge_counts[rel] = len(edges.loc[edges.relation == rel])

# Get set of all nodes
nodes = set()
for col in ['head', 'tail']:
    for node in edges[col].unique():
        nodes.add(node)

# Get node counts
total = len(nodes)
n_proteins = len([node for node in nodes if not node.startswith('C')]) # Protein IDs are just ints 
n_sideEffects = len([node for node in nodes if node.startswith('C') and not 'CID' in node]) # Side effects are of form C1234567
n_multidrugs = len([node for node in nodes if '-' in node]) # Multidrugs are of form 'CIDxxxx-CIDxxxx
n_drugs = len([node for node in nodes if node.startswith('CID')]) - n_multidrugs

# Create output dataframe
out_df = pd.DataFrame(columns=['type', 'name', 'count'])
out_df.loc[len(out_df)] = ['node', 'Total', total]
out_df.loc[len(out_df)] = ['node', 'Proteins', n_proteins]
out_df.loc[len(out_df)] = ['node', 'Side effects', n_sideEffects]
out_df.loc[len(out_df)] = ['node', 'Multidrugs', n_multidrugs]
out_df.loc[len(out_df)] = ['node', 'Drugs', n_drugs]

out_df.loc[len(out_df)] = ['edge', 'Total', len(edges)]
for count_name in edge_counts:
    out_df.loc[len(out_df)] = ['edge', count_name, edge_counts[count_name]]

# Write to disk
edgelist_name = file_name[:-4]
out_name = f'stats_{edgelist_name}.csv'
out_df.to_csv(out_name, index=False)
