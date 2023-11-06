import pandas as pd
import multiprocessing as mp
from itertools import product


def exist_check(triple, existing_list):
    if triple not in existing_list:
        return triple


if __name__ == '__main__':

    # Load ALL existing polypharmacy edges (training + holdout)
    poly_df = pd.read_csv(
        '../../../data/processed/polypharmacy/polypharmacy_edges.tsv',
        header=None, sep='\t'
    )

    # Get list of polypharmacy side effects
    side_effects = poly_df[1].unique()

    # Get list of drugs
    drugs = [
        entity for entity in
        pd.read_csv('entity_ids_selfloops.tsv', header=None, sep='\t')[1]
        if entity.startswith('CID')
    ]

    # Find non-existing edges and save
    existing_edges = poly_df.to_numpy().tolist()
    del poly_df

    all_possible_edges = product(*[drugs, side_effects, drugs])

    mp_args = [[triple, existing_edges] for triple in all_possible_edges]
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.starmap(exist_check, mp_args)

    # Filter out 'None' returned values
    non_existing_edges = [val for val in results if val]

    # Save
    pd.DataFrame(non_existing_edges).to_csv(
        'non_existing_polypharmacy_edges.tsv',
        header=None, sep='\t', index=False
    )
