from datetime import datetime
import os
import subprocess


def parse_time(timeStr):
    if timeStr[10] == 'T':
        time = datetime.strptime(timeStr, '%Y-%m-%dT%H:%M:%S')
    else:
        try:
            time = datetime.strptime(timeStr, '%a %d %b %H:%M:%S %Z %Y')
        except ValueError:
            time = datetime.strptime(timeStr, '%a %b %d %H:%M:%S %Z %Y')
    return time


if __name__ == '__main__':

    # Open output file to write
    with open('experiment_runtimes.csv', 'w+') as f:
        f.write('Dataset,Model,Runtime(secs)\n')

        # Iterate through experiments
        for dataset in ['selfloops', 'non-naive']:
            for model in ['complex', 'distmult', 'simple']:
                log_dir = f'{dataset}/{model}/output_logs'
                files = os.listdir(log_dir)

                # Iterate through output files of experiment
                runtime = 0
                for log in files:

                    # Parse start time
                    head = str(
                        subprocess.check_output(
                            f'head {log_dir}/{log}', shell=True
                        )
                    )
                    starttime_str = head.split('\\n')[3]
                    starttime = parse_time(starttime_str)

                    # Parse end time
                    tail = str(
                        subprocess.check_output(
                            f'tail {log_dir}/{log}', shell=True
                        )
                    )
                    cancelled_ind = tail.find('CANCELLED AT')
                    if cancelled_ind != -1:
                        endtime_str = tail[
                            cancelled_ind + 13: cancelled_ind + 32
                        ]
                    else:
                        endtime_str = tail[-31:-3]
                    endtime = parse_time(endtime_str)

                    # Calculate job runtime and add to experiment timer
                    timediff = endtime - starttime
                    runtime += timediff.total_seconds()

                # Save observations per experiment
                f.write(f'{dataset},{model},{runtime}\n')
