import pandas as pd
import numpy as np
from SALib.sample import saltelli
from SALib.analyze import sobol
from sklearn.linear_model import LinearRegression

raw_df = pd.read_csv('sobol_data.csv')

# Filter for best dataset
best_dataset = raw_df.sort_values('AUPRC')['dataset'].iloc[0]
df = raw_df.loc[raw_df.dataset == best_dataset]

# Prepare X and y
X = df[df.columns[7:]]
y = df[['AUROC', 'AUPRC', 'AP@50']]

# Fit OLS
model = LinearRegression().fit(X, y)

# Define sobol problem
problem = {
    'num_vars': len(X.columns),
    'names': X.columns,
    'bounds': [
        [min(X[col]), max(X[col])] for col in X.columns
    ]
}

# Sample the problem space 
param_values = saltelli.sample(problem, 1024)

# Get column indices of dummy vars
dummy_groups = {}
for col in X.columns:
    split = col.split('=')
    if len(split) > 1:
        variable = split[0]
        if variable not in dummy_groups:
            dummy_groups[variable] = [col]
        else:
            dummy_groups[variable].append(col)

# Perform argmax on dummy variables so that they are realistic inputs
def argmax_dummies(row):
    for variable in dummy_groups:
        dummies = dummy_groups[variable]
        values = row[dummies]
        max_val = max(values)
        for dummy in dummies:
            if row[dummy] < max_val:
                row[dummy] = 0
            else:
                row[dummy] = 1


param_values_df = pd.DataFrame(param_values, columns=X.columns)
param_values_df.apply(argmax_dummies, axis=1)  # Will take some time (~5mins) to run

# Also round the bool column sapled values to 0 or 1
param_values_df['lookup_embedder.regularize_args.weighted'] = round(
    param_values_df['lookup_embedder.regularize_args.weighted']
)

# Get model scores for sampled inputs
param_values = param_values_df.to_numpy()
preds = model.predict(param_values)

# Get sensitivity indices
auprc = [pred[1] for pred in preds]
Si = sobol.analyze(problem, np.array(auprc))

# Save
pd.DataFrame(zip(X.columns, Si['S1']), columns=['input', 'sensitivity']).to_csv('indices/first_order.csv', index=False)
pd.DataFrame(Si['S2'], columns=X.columns, index=X.columns).to_csv('indices/second_order.csv')
pd.DataFrame(zip(X.columns, Si['ST']), columns=['input', 'sensitivity']).to_csv('indices/total_order.csv', index=False)
