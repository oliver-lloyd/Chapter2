import argparse
from os import listdir

parser = argparse.ArgumentParser()
parser.add_argument('raw_experiment_dirs', nargs='+')
args = parser.parse_args()

out_counts = []
for experiment_path in args.raw_experiment_dirs:

    # Get experiment info
    exp_name = experiment_path.split('/')[-1]
    model_dataset = exp_name[16:]
    model = model_dataset.split('_')[0]
    dataset = model_dataset.split('_')[1]

    # Count sum of trials' epochs
    trial_folders = [
        loc for loc in listdir(experiment_path)
        if loc.startswith('0')
    ]
    total_epochs = 0
    for trial in trial_folders:
        checkpoints = [
            f for f in listdir(f'{experiment_path}/{trial}')
            if f.endswith('.pt') and 'best' not in f
        ]
        checkpoint_nums = [int(f.split('_')[-1][:-3]) for f in checkpoints]
        n_epochs = max(checkpoint_nums)
        total_epochs += n_epochs

    # Store for writing
    out_counts.append([model, dataset, total_epochs])

# Write to disk
with open('experiment_epochs.csv', 'w+') as out:
    out.write('model,dataset,total_epochs\n')
    for lst in out_counts:
        out.write(f'{lst[0]},{lst[1]},{lst[2]}\n')
