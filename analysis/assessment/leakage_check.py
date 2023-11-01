import pandas as pd
import multiprocessing as mp
from datetime import datetime


def direct_leakage_check(train_data, holdout_data, relation):
    train_data = train_data.loc[
        train_data[1] == relation
    ].to_numpy().tolist()
    holdout_data = holdout_data.loc[
        holdout_data[1] == relation
    ].to_numpy().tolist()
    check_bools = [edge in train_data for edge in holdout_data]
    if not any(check_bools):
        return False
    else:
        return relation


def polypharmacy_leakage_check(holdout_data, monopharmacy_dict, relation):
    holdout_data = holdout_data.loc[
        holdout_data[1] == relation
    ].to_numpy().tolist()
    check_bools = []
    for edge in holdout_data:
        head_leak = relation in monopharmacy_dict[edge[0]]
        tail_leak = relation in monopharmacy_dict[edge[2]]
        check_bools.append(head_leak or tail_leak)
    if not any(check_bools):
        return False
    else:
        return relation


if __name__ == '__main__':

    # Load holdout and training data
    holdout = pd.read_csv(
        '../../data/processed/polypharmacy/holdout_polypharmacy.tsv',
        header=None, sep='\t', nrows=10000
    )
    train = pd.DataFrame()
    for split in ['train', 'test', 'valid']:
        edges = pd.read_csv(
            f'../../../kge/data/non-naive/{split}.txt',
            header=None, sep='\t'
        )
        train = train.append(edges)
    train.reset_index(inplace=True, drop=True)

    output_report = []

    # Check 1) direct leakage
    mp_args1 = [[train, holdout, rel] for rel in holdout[1].unique()]
    with mp.Pool(mp.cpu_count()) as pool:
        direct_leakage = pool.starmap(direct_leakage_check, mp_args1)

    if any(direct_leakage):
        leaking_rels = [val for val in direct_leakage if val]
        output_report.append('Found direct leakage - holdout contains edges from training set.')
        output_report.append('Offending relations:')
        output_report.append(leaking_rels)
        output_report.append('\n')
    else:
        output_report.append('Did not find any direct leakage.')

    # Check 2) leakage via non-true polypharmacy
    # I.e. is the side effect already associated with one drug in the pair
    monopharmacy_edgelist = pd.read_csv(
        '../../data/processed/monopharmacy_edges.tsv',
        sep='\t', header=None
    )
    monopharmacy = {}
    for i, row in monopharmacy_edgelist.iterrows():
        drug = row[0]
        side_effect = row[2]
        if drug not in monopharmacy:
            monopharmacy[drug] = [side_effect]
        else:
            monopharmacy[drug].append(side_effect)
    holdout_drugs = holdout_drugs = set(holdout[0]).union(set(holdout[2]))
    for drug in holdout_drugs:
        if drug not in monopharmacy:
            # A few drugs exist in holdout data..
            # but we have no monopharmacy side effect data for them..
            # so add as empty list to avoid error
            monopharmacy[drug] = []

    mp_args2 = [[holdout, monopharmacy, rel] for rel in holdout[1].unique()]
    with mp.Pool(mp.cpu_count()) as pool:
        indirect_leakage = pool.starmap(polypharmacy_leakage_check, mp_args2)

    if any(indirect_leakage):
        leaking_rels_indirect = [val for val in indirect_leakage if val]
        output_report.append('Found leakage due to non-true polypharmacy in holdout edges.')
        output_report.append('Offending relations:')
        output_report.append(leaking_rels_indirect)
        output_report.append('\n')
    else:
        output_report.append('Did not find leakage due to non-true polypharmacy.')

    # Check 3) leakage via poly-/monopharmacy overlapping side effects
    mono_side_effects = set(monopharmacy_edgelist[2].unique())
    poly_side_effects = set(holdout[1].unique())
    overlapping = mono_side_effects.intersection(poly_side_effects)
    if not overlapping:
        output_report.append('Did not find leakage via overlapping mono-/polypharmacy side effects.')
    else:
        output_report.append('Found indirect leakage via overlapping mono-/polypharmacy side effects')
        output_report.append('Offending relations:')
        output_report.append(list(overlapping))
        output_report.append('\n')

    # Save output
    time = datetime.now()
    output_report.append(f'Data leakage report generated at {time}')
    with open(f'leakage_report.txt', 'w+') as f:
        for line in output_report:
            f.write(str(line) + '\n')
