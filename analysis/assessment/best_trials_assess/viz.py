import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from os import listdir

# Dict to convert strings
name_format = {
    'complex': 'ComplEx',
    'distmult': 'DistMult',
    'simple': 'SimplE',
    'selfloops': 'Selfloops',
    'non-naive': 'Non-Naive',
    'untrained': 'Untrained',
    'AUROC': 'Area Under Receiver Operating Characteristic',
    'AUPRC': 'Area Under Precision Recall Curve',
    'AP@50': 'Average Precision at 50'
}

# Load result data
results = pd.DataFrame()
for loc in listdir():
    if loc.startswith('2023'):
        df = pd.read_csv(f'{loc}/results_full.csv')
        name = loc[16:]
        model, dataset = name.split('_')
        df['model'] = model
        df['dataset'] = dataset
        results = results.append(df)

# Remove untrained model scores if not desired
include_untrained = False
if not include_untrained:
    results = results.loc[results.model != 'Untrained']

# Set drawing parameters
xmin = -0.5
xmax = 3.5
if not include_untrained:
    xmax -= 1
ymin = 0.8
ymax = 1
decagon_scores = {
    'AUROC': 0.872,
    'AUPRC': 0.832,
    'AP@50': 0.803
}

# Draw performance results
plot_other_scores = False
for metric in results.columns[1:4]:
    sns.boxplot(x='model', y=metric, hue='dataset', data=results)
    plt.hlines(
        decagon_scores[metric], xmin, xmax,
        label='Decagon', colors='black'
    )
    if plot_other_scores:
        if metric == 'AUROC':
            plt.hlines(0.998, xmin, xmax, label='Carletti\'s', linestyles='dashed', colors='red')
            plt.hlines(0.99, xmin, xmax, label='DSN-DDI', colors='green')
            plt.hlines(0.98, xmin, xmax, label='ADGCL', colors='blue')
        elif metric == 'AUPRC':
            plt.hlines(0.998, xmin, xmax, label='Carletti\'s', linestyles='dashed', colors='red')
            plt.hlines(0.983, xmin, xmax, label='MS-ADR', colors='green')
            plt.hlines(0.98, xmin, xmax, label='ADGCL', colors='blue')

    plt.xlim(xmin, xmax)
    plt.ylim(0.5 if metric == "AP@50" else ymin, ymax)
    plt.legend()
    plt.title(f'{metric} by model and dataset')
    plt.ylabel(name_format[metric])
    plt.xlabel('Model')
    plt.savefig(f'figures/{metric}.png')
    plt.clf()

# Load runtime data
epochs = pd.read_csv('../../experiments/experiment_epochs.csv')
runtimes = pd.read_csv('../../experiments/experiment_runtimes.csv')
eff_df = runtimes.merge(
    epochs, how='left',
    left_on=['Dataset', 'Model'],
    right_on=['dataset', 'model']
)
eff_df['secs_per_epoch'] = eff_df['Runtime(secs)'] / eff_df['total_epochs']
eff_df['model'] = [name_format[val] for val in eff_df['model'].values]
eff_df['dataset'] = [name_format[val] for val in eff_df['dataset'].values]

# Draw runtime barplot
sns.barplot(y='secs_per_epoch', x='model', hue='dataset', data=eff_df)
plt.ylim(0, 250)
plt.legend(title="Dataset")
plt.title('Per Epoch Running Time')
plt.ylabel('Seconds/Epoch')
plt.xlabel('Model')
plt.savefig('figures/runtime_per_epoch.png')
plt.clf()
