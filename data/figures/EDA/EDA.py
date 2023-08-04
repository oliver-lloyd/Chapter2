import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

non_naive_stats = pd.read_csv(
    '../../graphs/non-naive/stats_full_edgelist_non-naive.csv'
)

ignore_edges = ['Total', 'ProteinProteinInteraction', 'DrugTarget']
query = "type == 'edge' and name not in @ignore_edges"
non_naive_poly = non_naive_stats.query(query).reset_index(drop=True)
selfloop_stats = pd.read_csv(
    '../../graphs/selfloops/stats_full_edgelist_selfloops.csv'
)

mono_side_effects = pd.read_csv('../../raw/bio-decagon-mono.csv')
mono_side_effects = mono_side_effects['Individual Side Effect'].unique()
ignore_edges += list(mono_side_effects)

selfloop_poly = selfloop_stats.query(query).reset_index(drop=True)

# Assert that selfloop and non-naive have the same polypharmacy edge data
for i, row in selfloop_poly.iterrows():
    checks = row == non_naive_poly.loc[i]
    assert all(checks)
poly_stats = selfloop_poly

# Side effect edge frequency distribution
plt.clf()
sns.displot(poly_stats['count'], stat='proportion')
plt.xlabel('Count')
plt.ylabel('Proportion of Side Effects')
plt.title('Distribution of Side Effect Edge Counts')
plt.tight_layout()
plt.savefig('side_effect_freq.png')

# Graph components per side effect
plt.clf()
sns.displot(poly_stats['num_components'], stat='proportion')
plt.xlabel('Number of Components')
plt.ylabel('Proportion of Side Effects')
plt.title('Distribution of Side Effect Component Count')
plt.tight_layout()
plt.savefig('side_effect_components.png')

# Component sizes per side effect
plt.clf()
sns.displot(poly_stats['largest_component_diameter'], stat='proportion')
plt.xlabel('Diameter of Biggest Component')
plt.ylabel('Proportion of Side Effects')
plt.title('Distribution of Largest Component Diameter per Side Effect')
plt.tight_layout()
plt.savefig('side_effect_component_size.png')

# Clustering coef per side effect
plt.clf()
sns.displot(poly_stats['transitivity'], stat='proportion')
plt.xlabel('Transitivity')
plt.ylabel('Proportion of Side Effects')
plt.title('Side Effect Transitivity Distribution')
plt.tight_layout()
plt.savefig('side_effect_clustering.png')
