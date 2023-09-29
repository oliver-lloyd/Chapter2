import pandas as pd
import argparse
from os import system


def get_nodelist(df):
    heads = df[0].unique()
    tails = df[2].unique()
    nodes = set(heads).intersection(set(tails))
    return nodes


def node_check(node_name):
    try:
        int(node_name)  # Gene IDs are just ints
        return True
    except ValueError:
        if node_name.startswith('CID'):
            try:
                int(node_name[3:])  # Drug IDs are 'CID' + 9 digits
                return True
            except ValueError:
                print(node_name)
                return False
        else:
            print(node_name)
            return False


def edge_check(edge_name):
    if edge_name in ['ProteinProteinInteraction', 'DrugTarget']:
        return True
    elif edge_name.startswith('C'):
        try:
            int(edge_name[1:])  # Side effect IDs are 'C' + 7 digits
            check = True
        except ValueError:
            check = False
        return check
    else:
        return False


if __name__ == '__main__':

    # Load data
    selfloops = pd.read_csv(
        'selfloops/edgelist_selfloops.tsv',
        header=None, sep='\t',
        dtype={0: str, 1: str, 2: str}
    )
    nonnaive = pd.read_csv(
        'non-naive/edgelist_non-naive.tsv',
        header=None, sep='\t',
        dtype={0: str, 1: str, 2: str}
    )
    holdout = pd.read_csv(
        '../processed/polypharmacy/holdout_polypharmacy.tsv',
        header=None, sep='\t',
        dtype={0: str, 1: str, 2: str}
    )

    # Check for reasonable edgelist lengths
    n_monopharma = len(
        pd.read_csv('../processed/monopharmacy_edges.tsv', header=None)
    )
    assert len(selfloops) == len(nonnaive) + n_monopharma
    assert len(holdout) < 0.1 * len(nonnaive)

    # Check for duplicate edges
    assert len(selfloops) == len(selfloops.drop_duplicates())
    assert len(nonnaive) == len(nonnaive.drop_duplicates())

    # Iterative checks
    for df in [selfloops, nonnaive, holdout]:

        # Check meta edge IDs
        for edge in df[1].unique():
            assert edge_check(edge)

        # Check node IDs
        nodes = get_nodelist(df)
        for node in nodes:
            assert node_check(node)

    print('All tests passed.')
