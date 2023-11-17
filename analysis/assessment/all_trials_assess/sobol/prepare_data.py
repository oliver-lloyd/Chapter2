import pandas as pd
import numpy as np
from os import listdir

# Load assessment results
base_dir = '..'
result_summary = []
for loc in listdir(base_dir):
    if loc.startswith('2023') and 'best_simple_selfloops' not in loc:
        exp_dir = base_dir + f'/{loc}'
        trial_nums = [x for x in listdir(exp_dir) if x.startswith('0')]
        for trial_num in trial_nums:
            trial_dir = exp_dir +f'/{trial_num}'
            trial_results = pd.read_csv(f'{trial_dir}/results_full.csv')
            auroc = np.median(trial_results['AUROC'])
            auprc = np.median(trial_results['AUPRC'])
            ap50 = np.median(trial_results['AP@50'])
            result_summary.append(
                [loc, trial_num, auroc, auprc, ap50]
            )
result_summary = pd.DataFrame(
    result_summary,
    columns = ['experiment', 'trial', 'AUROC', 'AUPRC', 'AP@50']
)

# Load experiment traces
experiment_traces = pd.DataFrame()
base_dir = '../../../experiments'
for graph in ['non-naive', 'selfloops']:
    for model in ['complex', 'distmult', 'simple']:
        target_dir = f'{base_dir}/{graph}/{model}'
        files = listdir(target_dir)
        trace_file = [f for f in files if f.startswith('2023') and f.endswith('.csv')][0]
        trace = pd.read_csv(f'{target_dir}/{trace_file}')
        trace['experiment'] = trace_file.split('.')[0]
        trace['trial'] = [str(num).zfill(5) for num in trace.child_folder]
        experiment_traces = experiment_traces.append(trace)

# Merge datasets
sobol_df = result_summary.merge(experiment_traces, on=['experiment', 'trial'])

# Merge 'model.parameter' columns
models = sobol_df.model.unique()
to_merge_cols = [col for col in sobol_df.columns if col.split('.')[0] in models]
for col in to_merge_cols:
    model = col.split('.')[0]
    merged_col_name = col[len(model) + 1: ]
    if merged_col_name not in sobol_df.columns:
        sobol_df[merged_col_name] = 0.0
    value_series = sobol_df[col].loc[pd.notna(sobol_df[col])]
    for ind in value_series.index:
        sobol_df[merged_col_name].loc[ind] = value_series[ind]

# Remove unneeded columns
drop_cols = [
    'job_id', 'reciprocal', 'job', 'split', 'epoch', 
    'avg_loss', 'avg_penalty', 'avg_loss', 'avg_cost',
    'metric_name', 'metric',
    'child_folder', 'child_job_id', 'lookup_embedder.regularize'
] + to_merge_cols
keep_cols = [col for col in sobol_df.columns if col not in drop_cols]
sobol_df = sobol_df[keep_cols]

# Dummify categorical hyperparams (and cast bool to int)
for col in sobol_df.columns[7:]:
    dtype = sobol_df.dtypes[col]
    if dtype == object:
        dummy_df = pd.get_dummies(sobol_df.pop(col))
        dummy_df.columns = [f'{col}={value}' for value in dummy_df.columns]
        sobol_df = pd.concat([sobol_df, dummy_df], axis=1)
        if len(sobol_df) > 600:
            break
    elif dtype == bool:
        sobol_df[col] = sobol_df[col].astype(int)

# Save
sobol_df.to_csv('sobol_data.csv', index=False)