import pandas as pd
import numpy as np
import multiprocessing as mp
import torch
import decagon_rank_metrics
import argparse
from kge.model import KgeModel
from kge.util.io import load_checkpoint
from sklearn.metrics import roc_auc_score, average_precision_score
from copy import copy


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
parser.add_argument('--num_cores')
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

# Convert holdout edges to IDs if necessary
if all(holdout.dtypes == object):
    entity_name_to_id = {entity_ids[key]: key for key in entity_ids}
    holdout[0] = [entity_name_to_id[name] for name in holdout[0]]
    holdout[2] = [entity_name_to_id[name] for name in holdout[2]]

    relation_name_to_id = {relation_ids[key]: key for key in relation_ids}
    holdout[1] = [relation_name_to_id[name] for name in holdout[1]]
elif any(holdout.dtypes == object):
    raise ValueError(
        'Appears as though there is a mix of IDs and strings in holdout data.'
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
# TODO: why does 2.1 never print? Something goes wrong when predicting from model in subprocess
def calculate_metrics_for_rel(rel_id, holdout_this_rel):
    print(1)
    # Get data
    positive_edges = holdout_this_rel.to_numpy().tolist()
    train_subdf = full_edgelist.loc[full_edgelist[1] == rel_id]
    negative_edges = create_negative_edges(
        positive_edges + train_subdf.to_numpy().tolist(),
        list(entity_ids.keys())
    )
    edges_to_score = positive_edges + negative_edges
    s = torch.Tensor([edge[0] for edge in edges_to_score])
    p = torch.Tensor([rel_id for edge in edges_to_score])
    o = torch.Tensor([edge[2] for edge in edges_to_score])
    print(2)
    # Get predictions
    if model.model != 'reciprocal_relations_model':
        print(2.1)
        preds = model.score_spo(s, p, o).tolist()
        print(2.2)
    else:
        preds_s = model.score_spo(s, p, o, direction='s').tolist()
        preds_o = model.score_spo(s, p, o, direction='o').tolist()
        preds = [np.mean(tup) for tup in zip(preds_s, preds_o)]
    labels = [1 for _ in positive_edges] + [0 for _ in negative_edges]
    assert len(preds) == len(labels)
    print(3)
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
    print(4)
    return [relation_ids[rel_id], roc, prc, ap50]


# Perform calculations in parallel
mp_args = list(holdout.groupby(1))
with mp.Pool(int(args.num_cores)) as pool:
    results = pool.starmap(calculate_metrics_for_rel, mp_args)

# Store results
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
