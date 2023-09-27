import argparse
import pandas as pd
import numpy as np
from os import system, listdir
from sklearn.model_selection import train_test_split

np.random.seed(0)
parser = argparse.ArgumentParser()
parser.add_argument('libkge_base_data', type=str)
args = parser.parse_args()

# Check if data already exists, create if not
for graph_name in ['selfloops', 'non-naive']:
    if graph_name in listdir(args.libkge_base_data) :
        print(f'Found directory {graph_name} in LibKGE/data. Skipping..')
    else:
        graph_dir = args.libkge_base_data + '/' + graph_name
        system(f'mkdir {graph_dir}')
        print(f'Created {graph_dir}')

        # Load graph edgelist AFTER holdout edges have been removed
        edgelist = pd.read_csv(
            f'{graph_name}/edgelist_{graph_name}.tsv', 
            sep='\t', header=None
        )

        # Split off test and validation sets, ensuring they are same size
        split_size = 0.1
        train_valid, test = train_test_split(edgelist, test_size=split_size)
        train, valid = train_test_split(train_valid, test_size=split_size/(1-split_size))
        assert np.abs(len(valid) - len(test)) <= 1

        # Save data splits to disk
        train.to_csv(
            f'{graph_dir}/train.txt', 
            header=None, index=False, sep='\t'
        )
        valid.to_csv(
            f'{graph_dir}/valid.txt', 
            header=None, index=False, sep='\t'
        )
        test.to_csv(
            f'{graph_dir}/test.txt', 
            header=None, index=False, sep='\t'
        )
        print(f'Moved train splits to {graph_dir}')

        # Process with LibKGE
        system(
            f'python {args.libkge_base_data}/preprocess/preprocess_default.py {graph_dir}'
        )
        print(f'Finished processing {graph_dir}')