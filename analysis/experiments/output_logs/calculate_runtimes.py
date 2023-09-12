from datetime import datetime
import os

"""
Will be good to do a sanity check that scripts are running for 168 or
250 hours.
"""


def get_start_time(file_head):
    with open(file_head) as f:
        contents = f.read()
    raw_datetime = contents.split('\n')[3][4:]
    try:
        time = datetime.strptime(raw_datetime, '%b %d %H:%M:%S BST %Y')
    except ValueError:
        time = datetime.strptime(raw_datetime, '%d %b %H:%M:%S BST %Y')

    return time


def get_end_time(file_tail):
    search_string = 'CANCELLED AT'
    with open(file_tail) as f:
        contents = f.read()
        cancelled_at = [
            line for line in contents.split('\n')
            if search_string in line
        ]

    if len(cancelled_at) > 0:
        # Case when Slurm cancelled job
        raw_line = cancelled_at[0]
        cancel_index = raw_line.index(search_string)
        raw_datetime = raw_line[cancel_index + len(search_string) + 1: -22]
        time = datetime.strptime(raw_datetime, '%Y-%m-%dT%H:%M:%S')
    else:
        # Case when job ended naturally within time limit
        raw_line = contents.split('\n')[-2]
        raw_datetime = raw_line[4:]
        try:
            time = datetime.strptime(raw_datetime, '%b %d %H:%M:%S BST %Y')
        except ValueError:
            time = datetime.strptime(raw_datetime, '%d %b %H:%M:%S BST %Y')

    return time


with open('runtimes.csv', 'w+') as f:
    f.write('job_id,seconds\n')
    for slurm_dir in os.listdir():
        if slurm_dir.startswith('slurm'):
            start = get_start_time(f'{slurm_dir}/head_{slurm_dir}.out')
            end = get_end_time(f'{slurm_dir}/tail_{slurm_dir}.out')
            runtime = end - start
            f.write(f'{slurm_dir},{runtime.total_seconds()}\n')
