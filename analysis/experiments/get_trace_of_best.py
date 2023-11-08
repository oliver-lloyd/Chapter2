import pandas as pd
from os import listdir

out_df = pd.DataFrame()
for dataset in ['non-naive', 'selfloops']:
    for model in listdir(dataset):
        if model != 'feature_vectors':
            path = f'{dataset}/{model}'
            trace_file = [f for f in listdir(path) if f.startswith('2') and f.endswith('.csv')][0]
            trace_df = pd.read_csv(f'{path}/{trace_file}')
            max_metric = trace_df.loc[trace_df.metric == max(trace_df.metric)]
            assert len(max_metric) == 1
            max_metric['dataset'] = dataset
            out_df = out_df.append(max_metric)

out_df.to_csv('best_traces.csv', index=False)