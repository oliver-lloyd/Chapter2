import pandas as pd
import numpy as np
import torch
from sklearn.decomposition import PCA

# Load data
edges = pd.read_csv('../monopharmacy_edges.tsv', header=None, sep='\t').drop(columns=[1])
edges.columns = ['drug', 'side_effect']

# Create dataframe for full features
drugs = edges.drug.unique()
side_effects = edges.side_effect.unique()
output_shape = (len(drugs), len(side_effects))
full_features = pd.DataFrame(np.zeros(output_shape), index=drugs, columns=side_effects, dtype=int)

# Populate full features
for drug, sub_df in edges.groupby('drug'):
    full_features.loc[drug][sub_df.side_effect] = 1

for n_dim in [32, 64, 128, 256]:
    pca = PCA(n_components=n_dim)
    new_matrix = pca.fit_transform(full_features)
    torch.save(torch.tensor(new_matrix), f'{n_dim}dim.pt')
