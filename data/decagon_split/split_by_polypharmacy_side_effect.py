import pandas as pd
import argparse
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('full_edgelist')
parser.add_argument('--out_dir')
args = parser.parse_args()

# Load target edgelist
edges = pd.read_csv(args.full_edgelist, header=None, sep='\t', dtype={0:str, 1:str, 2:str})

# Get list of polypharmacy side effects
poly_edges = pd.read_csv('../raw/bio-decagon-combo.csv')['Polypharmacy Side Effect'].unique()

# Create holdout data that has 10% of each
train_df = pd.DataFrame()
holdout_df = pd.DataFrame()
for edge_type, subdf in edges.groupby(1):
    if edge_type in poly_edges:
        train_edges, test_edges = train_test_split(subdf, test_size=0.1)
        train_df = train_df.append(train_edges)
        holdout_df = holdout_df.append(test_edges)
    else:
        train_df = train_df.append(subdf)

# Save
if args.full_edgelist.endswith('/'):
    args.full_edgelist = args.full_edgelist[:-1]
filename = args.full_edgelist.split('/')[-1]

if args.out_dir:
    train_outname = f'{args.out_dir}/train_{filename}'
    holdout_outname = f'{args.out_dir}/holdout_{filename}'
else:
    train_outname = f'train_{filename}'
    holdout_outname = f'holdout_{filename}'

train_df.to_csv(train_outname, header=None, index=False, sep='\t')
holdout_df.to_csv(holdout_outname, header=None, index=False, sep='\t')