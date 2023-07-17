import pandas as pd
import numpy as np
import torch
from sklearn.decomposition import PCA

# Load data
edges = pd.read_csv('../../../../data/processed/monopharmacy_edges.tsv', header=None, sep='\t').drop(columns=[1])
edges.columns = ['drug', 'side_effect']

# Load nodelist to function as index
nodelist = pd.read_csv('entity_ids.del', header=None, sep='\t', index_col=0)

# Create dataframe for full features
drugs = edges.drug.unique()
side_effects = edges.side_effect.unique()
output_shape = (len(nodelist), len(side_effects))
full_features = pd.DataFrame(np.zeros(output_shape), index=nodelist[1], columns=side_effects, dtype=int)

# Populate features of drugs
for drug, sub_df in edges.groupby('drug'):
    full_features.loc[drug][sub_df.side_effect.values] = 1

# Create unique one-hot vectors for all non-drug nodes
non_drugs = [node for node in nodelist[1] if node not in drugs]
feature_count = len(full_features.columns)
assert 1 < min([sum(row) for i, row in full_features.loc[drugs].iterrows()])  # Make sure drug rows have at least 3 non-zeros to avoid duplicates with one-/two-hot vectors created below
unique_two_hots = []
for i, nondrug in enumerate(non_drugs):
    if i < feature_count:
        hot_col = full_features.columns[i]
        full_features.loc[nondrug][hot_col] = 1
    else:
        j = i - feature_count + 1
        k = j
        while k == j or {k, j} in unique_two_hots:  # Check if chosen value of k causes duplicates
            k = np.random.choice(feature_count)
        unique_two_hots.append({k, j})
        hot_cols = full_features.columns[[j, k]]
        full_features.loc[nondrug][hot_cols] = 1

# Save full feature set
full_features.to_csv('full_features.csv', index=False)

# Iteratively reduce dimensions and save
for n_dim in [32, 64, 128, 256, 512]:
    pca = PCA(n_components=n_dim)
    new_matrix = pca.fit_transform(full_features)
    torch.save(torch.tensor(new_matrix), f'{n_dim}dim.pt')
