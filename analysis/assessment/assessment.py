import pandas as pd
import numpy as np
import torch
import decagon_rank_metrics
import argparse
from kge.model import KgeModel
from kge.util.io import load_checkpoint
from sklearn.metrics import roc_auc_score, average_precision_score


def create_negative_edges(pos_edges, entity_list):
    rel = pos_edges[0][1]
    rel_check = all([edge[1] == rel for edge in pos_edges])
    if not rel_check:
        raise ValueError('Positive edges contains multiple relation types.')
    neg_edges = []
    while len(neg_edges) < len(pos_edges):
        head = np.random.choice(entity_list)
        tail = np.random.choice(entity_list)
        edge = [head, rel, tail]
        if edge not in pos_edges and edge not in neg_edges:
            neg_edges.append(edge)
    return neg_edges


# Get user args
parser = argparse.ArgumentParser()
parser.add_argument('model_checkpoint')
parser.add_argument('holdout_edges')
parser.add_argument('libkge_data_dir')
parser.add_argument('--out_name')
args = parser.parse_args()

# Load data
entity_ids = pd.read_csv(
    f'{args.libkge_data_dir}/entity_ids.del',
    sep='\t',
    header=None,
    index_col=0
).to_dict()[1]
relation_ids = pd.read_csv(
    f'{args.libkge_data_dir}/relation_ids.del',
    sep='\t',
    header=None,
    index_col=0
).to_dict()[1]
full_edgelist = pd.DataFrame()
for split in ['train', 'test', 'valid']:
    full_edgelist = full_edgelist.append(pd.read_csv(
        f'{args.libkge_data_dir}/{split}.del',
        sep='\t',
        header=None
    ))
holdout = pd.read_csv(
    args.holdout_edges,
    sep='\t',
    header=None
)

# Load checkpoint
checkpoint = load_checkpoint(args.model_checkpoint)
model = KgeModel.create_from(checkpoint)
model_name = model.model
if model_name == 'reciprocal_relations_model':
    model_name = model.config.options.get(
        'reciprocal_relations_model'
        )['base_model']['type']

# Calculate metrics per relation type
# TODO: parallelise this, but beware of memory usage on large data
results = []
for rel_id, subdf in holdout.groupby(1):
    # Get data
    positive_edges = subdf.to_numpy().tolist()
    train_subdf = full_edgelist.loc[full_edgelist[1] == rel_id]
    negative_edges = create_negative_edges(
        positive_edges + train_subdf.to_numpy().tolist(),
        list(entity_ids.keys())
    )
    edges_to_score = positive_edges + negative_edges
    s = torch.Tensor([edge[0] for edge in edges_to_score])
    p = torch.Tensor([rel_id for edge in edges_to_score])
    o = torch.Tensor([edge[2] for edge in edges_to_score])

    # Get predictions
    if model.model != 'reciprocal_relations_model':
        preds = model.score_spo(s, p, o).tolist()
    else:
        preds_s = model.score_spo(s, p, o, direction='s').tolist()
        preds_o = model.score_spo(s, p, o, direction='o').tolist()
        preds = [np.mean(tup) for tup in zip(preds_s, preds_o)]
    labels = [1 for _ in positive_edges] + [0 for _ in negative_edges]
    assert len(preds) == len(labels)

    # Calculate area-under metrics
    roc = roc_auc_score(labels, preds)
    prc = average_precision_score(labels, preds)

    # Calculate average precision @ 50 using Decagon's function
    edges_ranked = pd.DataFrame(zip(preds, edges_to_score))
    edges_ranked.sort_values(0, ascending=False, inplace=True)
    ap50 = decagon_rank_metrics.apk(
        positive_edges,
        edges_ranked[1].values,
        k=50
    )

    # Store metrics for target relation
    relation = relation_ids[rel_id]
    results.append([relation, roc, prc, ap50])

results = pd.DataFrame(
    results,
    columns=['Relation', 'AUROC', 'AUPRC', 'AP@50']
)
if args.out_name:
    results.to_csv(args.out_name, index=False)
else:
    data_dir_name = args.libkge_data_dir
    if data_dir_name.endswith('/'):
        data_dir_name = data_dir_name[:-1]
    data_dir_name = data_dir_name.split('/')[-1]
    results.to_csv(f'results_{data_dir_name}_{model_name}.csv', index=False)
