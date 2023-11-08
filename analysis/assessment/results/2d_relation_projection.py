import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from kge.model import KgeModel
from kge.util.io import load_checkpoint

# Load embeddings
checkpoint_path = '../../experiments/selfloops/complex/20230929-111626-complex_selfloops/00045/checkpoint_best.pt'
checkpoint = load_checkpoint(checkpoint_path)
checkpoint['config'].set('dataset.name', '/Users/fu19841/Documents/thesis_analysis/kge/data/selfloops')  # for local testing, delete later
libkge_data_dir = checkpoint['config'].get('dataset.name')
model = KgeModel.create_from(checkpoint)
all_embeds = model.get_p_embedder()._embeddings.state_dict()['weight']

# Get index of target rels
target_rels = pd.read_csv(
    '../../../data/processed/polypharmacy/holdout_polypharmacy.tsv',
    header=None, sep='\t'
)[1].unique()
rel_ids = pd.read_csv(
    f'{libkge_data_dir}/relation_ids.del',
    header=None, sep='\t', index_col=1
).to_dict()[0]
target_ids = [rel_ids[target] for target in target_rels]

# Select relevant embeddings
embeds = all_embeds[target_ids]

# Reduce to 50 dimensions with PCA, as per TSNE docs
embeds50 = PCA(50).fit_transform(embeds)

# Project into 2d
embeds2 = TSNE(2).fit_transform(embeds50)

# Get labels to recreate fig 4 from decagon paper
side_effects = pd.read_csv('../../../data/raw/bio-decagon-combo.csv')
side_effects.drop_duplicates(
    ['Polypharmacy Side Effect', 'Side Effect Name'],
    inplace=True
)
side_effect_lookup = dict(
    zip(
        side_effects['Polypharmacy Side Effect'],
        side_effects['Side Effect Name']
    )
)
decagon_fig4_groups = {
    'Uterine polyp': [
        'Uterine polyp',
        'postmenopausal bleeding',
        'breast dysplasia',
        'uterine bleeding',
    ],
    'Pancreatitis': [
        'pancreatitis',
        'cholelithiasis',
        'Diabetes',
        'abdominal pain',
    ],
    'Viral meningitis': [
        'Meningitis Viral',
        'encephalitis viral',
        'otitis media',
        'hypogammaglobulinaemia',
    ],
    'Thyroid disease': [
        'thyroid disease',
        'hypothyroid',
        'sleep apnea',
        'fibromyalgia'
    ]
}
rel_names = {rel_ids[name]: name for name in rel_ids}
target_codes = [rel_names[id] for id in target_ids]
target_names = [side_effect_lookup[code] for code in target_codes]
target_group = []
for se in target_names:
    found = False
    for group in decagon_fig4_groups:
        if se in decagon_fig4_groups[group]:
            target_group.append(group)
            found = True
            break
    if not found:
        target_group.append(None)

# Bring data together
plot_df = pd.DataFrame(embeds2)
plot_df['group'] = target_group

# Draw
plt.clf()
first_layer = plot_df.loc[pd.isna(plot_df.group)]
second_layer = plot_df.loc[pd.notna(plot_df.group)]
sns.scatterplot(first_layer, x=0, y=1, color='grey', alpha=0.5)
sns.scatterplot(second_layer, x=0, y=1, hue='group')
plt.legend(title='Co-occuring with:')
plt.xticks([])
plt.yticks([])
plt.xlabel('')
plt.ylabel('')
plt.title('t-SNE 2d projection of side effect embeddings')
plt.savefig('figures/tSNE_side_effects.png')
