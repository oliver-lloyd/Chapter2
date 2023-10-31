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
    if '.' not in loc:
        df = pd.read_csv(f'{loc}/results_full.csv')
        components = loc.split('_')
        df['model'] = name_format[components[0]]
        df['dataset'] = name_format[components[1]]
        results = results.append(df)

# Set drawing parameters
xmin = -0.5
xmax = 3.5
ymin = 0
ymax = 1
decagon_scores = {
    'AUROC': 0.872,
    'AUPRC': 0.832,
    'AP@50': 0.803
}

# Draw
for metric in results.columns[1:4]:
    sns.boxplot(x='model', y=metric, hue='dataset', data=results)
    plt.hlines(
        decagon_scores[metric], xmin, xmax,
        label=f'Decagon {metric}', colors='red'
    )
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.legend()
    plt.title(f'{metric} by model and dataset')
    plt.ylabel(name_format[metric])
    plt.xlabel('Model')
    plt.savefig(f'{metric}.png')
    plt.clf()
