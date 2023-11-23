import pandas as pd
import multiprocessing as mp
from itertools import combinations
from os import listdir


def dettmers_check(rel1_edges, full_edges, relation2):
    relation1 = rel1_edges.relation.iloc[0]
    rel2_edges = full_edges.loc[full_edges.relation == relation2]
    rel1_edges.columns = ['tail', 'relation', 'head']
    overlap = rel1_edges.merge(rel2_edges, on=['tail', 'head']).drop_duplicates()
    return [relation1, relation2, len(overlap)/len(rel1_edges)]


if __name__ == '__main__':

    # Load data
    edgelist = pd.read_csv(
        '../../data/graphs/non-naive/edgelist_non-naive.tsv',
        sep='\t', header=None
    )
    edgelist.columns = ['head', 'relation', 'tail']  # col names for script readability

    # Prepare/load output dataframe
    outfile = 'dettmers_proportions.csv'
    cols = ['relation1', 'relation2', 'inverse_proportion']
    if outfile in listdir():
        out_df = pd.read_csv(outfile)
    else:
        out_df = pd.DataFrame(
            columns=cols
        )

    # Iterate through relations
    relations = edgelist.relation.unique()
    for rel1 in relations:
        if rel1 not in out_df.relation1:
            edges1 = edgelist.loc[edgelist.relation == rel1]

            # Calculate inverse proportion with all other relations
            mp_args = [[edges1, edgelist, rel2] for rel2 in relations]
            with mp.Pool(mp.cpu_count()) as pool:
                mp_outputs = pool.starmap(dettmers_check, mp_args)

            # Save
            df = pd.DataFrame(mp_outputs, columns=cols)
            out_df = out_df.append(df)
            out_df.to_csv(outfile, index=False)
