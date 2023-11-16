import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from os import listdir

# Iterate through experiments and load data
experiments = [folder for folder in listdir() if folder.startswith('2023')]
summary_df = pd.DataFrame()
for exp in experiments:
    model_dataset = exp[16:]
    model, dataset = model_dataset.split('_')
    trial_performance = pd.DataFrame(
        columns=['dataset', 'model', 'trial', 'med_AUROC', 'med_AUPRC', 'med_AP50']
    )
    
    # Iterate through experiment trials and process data
    trials = [sub_dir for sub_dir in listdir(exp) if sub_dir.startswith('0')]
    for trial in trials:
        df = pd.read_csv(f'{exp}/{trial}/results_full.csv')
        auroc = np.median(df['AUROC'])
        auprc = np.median(df['AUPRC'])
        ap50 = np.median(df['AP@50'])
        trial_performance.loc[len(trial_performance)] = [
            dataset,
            model,
            int(trial),
            auroc,
            auprc,
            ap50
        ]

    # Get rolling maximums for the three metrics
    trial_performance.sort_values('trial', inplace=True)
    rolling_maximums = {}
    for i, row in trial_performance.iterrows():
        for metric in ['AUROC', 'AUPRC', 'AP50']:
            value = row[f'med_{metric}']
            if metric not in rolling_maximums:
                rolling_maximums[metric] = [value]
            else: 
                current_max = rolling_maximums[metric][-1]
                if value <= current_max:
                    rolling_maximums[metric].append(current_max)
                else:
                     rolling_maximums[metric].append(value)
    # Store rolling maximums in df
    for metric in rolling_maximums:
        trial_performance[f'rolling_max_{metric}'] = rolling_maximums[metric]

    # Append results to main dataframe
    summary_df = summary_df.append(trial_performance)

summary_df.reset_index(inplace=True, drop=True)

# Rolling AUROC per trial
plt.clf()
sns.lineplot(summary_df, x='trial', y='rolling_max_AUROC', hue='model', style='dataset')
plt.xlabel('Trial number')
plt.ylabel('Area Under Receiver Operating Characteristic')
plt.title('Rolling best AUROC against trial number')
ymin = 0.8
ymax = 1.01
plt.vlines(50, ymin=ymin, ymax=ymax, alpha=0.2)
plt.ylim(ymin, ymax)
plt.savefig('figures/rolling_AUROC.png')

# Rolling AUPRC per trial
plt.clf()
sns.lineplot(summary_df, x='trial', y='rolling_max_AUPRC', hue='model', style='dataset')
plt.xlabel('Trial number')
plt.ylabel('Area Under Precision-Recall Curve')
plt.title('Rolling best AUPRC against trial number')
ymin = 0.8
ymax = 1.01
plt.vlines(50, ymin=ymin, ymax=ymax, alpha=0.2)
plt.ylim(ymin, ymax)
plt.savefig('figures/rolling_AUPRC.png')

# Rolling AP50 per trial
plt.clf()
sns.lineplot(summary_df, x='trial', y='rolling_max_AP50', hue='model', style='dataset')
plt.xlabel('Trial number')
plt.ylabel('Average Precision at 50')
plt.title('Rolling best AP@50 against trial number')
ymin = 0.7
ymax = 1.01
plt.vlines(50, ymin=ymin, ymax=ymax, alpha=0.2)
plt.ylim(ymin, ymax)
plt.savefig('figures/rolling_AP@50.png')