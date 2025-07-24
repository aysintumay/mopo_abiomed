# Borrow a lot from tianshou:
# https://github.com/thu-ml/tianshou/blob/master/examples/mujoco/plotter.py
import csv
import os
import re

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import tqdm
import argparse

from tensorboard.backend.event_processing import event_accumulator


def convert_tfenvents_to_csv(root_dir, xlabel, ylabel, ylabel2):
    """Recursively convert test/metric from all tfevent file under root_dir to csv."""
    tfevent_files = []
    for dirname, _, files in os.walk(root_dir):
        for f in files:
            absolute_path = os.path.join(dirname, f)
            if re.match(re.compile(r"^.*tfevents.*$"), absolute_path):
                tfevent_files.append(absolute_path)
    print(f"Converting {len(tfevent_files)} tfevents files under {root_dir} ...")
    result = {}
    with tqdm.tqdm(tfevent_files) as t:
        for tfevent_file in t:
            t.set_postfix(file=tfevent_file)
            output_file = os.path.join(os.path.split(tfevent_file)[0], ylabel+'.csv')
            ea = event_accumulator.EventAccumulator(tfevent_file)
            ea.Reload()
            content = [[xlabel, ylabel, ylabel2]]
            for test_rew in ea.scalars.Items('eval/'+ylabel):
                content.append(
                    [
                        round(test_rew.step, 4),
                        round(test_rew.value, 4),
                    ]
                )
            # csv.writer(open(output_file, 'w')).writerows(content)
            

            for test_rew in ea.scalars.Items(ylabel2):
                content[1].extend([round(test_rew.value, 4)])
            csv.writer(open(output_file, 'w')).writerows(content)
            result[output_file] = content
    return result


def merge_csv(csv_files, root_dir, xlabel, ylabel, ylabel2):
    """Merge result in csv_files into a single csv file."""
    assert len(csv_files) > 0
    sorted_keys = sorted(csv_files.keys())
    sorted_values = [csv_files[k][1:] for k in sorted_keys]
    content = [
        [xlabel, ylabel+'_mean', ylabel+'_std', ylabel+'_2_mean', ylabel+'_2_std']
    ]
    for rows in zip(*sorted_values):
        array = np.array(rows)
        assert len(set(array[:, 0])) == 1, (set(array[:, 0]), array[:, 0])
        line = [rows[0][0], round(array[:, 1].mean(), 4), round(array[:, 1].std(), 4), 
               round(array[:, 2].mean(), 4), round(array[:, 2].std(), 4)]
        content.append(line)
    output_path = os.path.join(root_dir, ylabel+".csv")
    print(f"Output merged csv file to {output_path} with {len(content[1:])} lines.")
    csv.writer(open(output_path, "w")).writerows(content)


def csv2numpy(file_path):
    df = pd.read_csv(file_path)
    step = df.iloc[:,0].to_numpy()
    mean = df.iloc[:,1].to_numpy()
    std = df.iloc[:,2].to_numpy()
    return step, mean, std


def smooth(y, radius=0):
    convkernel = np.ones(2 * radius + 1)
    out = np.convolve(y, convkernel, mode='same') / np.convolve(np.ones_like(y), convkernel, mode='same')
    return out


def plot_figure(root_dir, task, algo_list, x_label, y_label, title, smooth_radius, color_list=None):
    fig, ax = plt.subplots()
    if color_list is None:
        color_list = [COLORS[i] for i in range(len(algo_list))]
    for i, algo in enumerate(algo_list):
        x, y, shaded = csv2numpy(os.path.join(root_dir, task, algo, y_label+'.csv'))
        # y = smooth(y, smooth_radius)
        # shaded = smooth(shaded, smooth_radius)
        # x=smooth(x, smooth_radius)
        ax.plot(x, y, color=color_list[i], label=algo_list[i])
        ax.fill_between(x, y-shaded, y+shaded, color=color_list[i], alpha=0.2)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='plotter')
    parser.add_argument(
        '--root-dir', 
        #default='log/hopper-medium-replay-v0/mopo',
         default='log', help='root dir'
    )
    parser.add_argument(
        '--task', default='halfcheetah-medium-v2', help='task'
    )
    parser.add_argument(
        '--algos', default=["mopo"], help='algos'
    )
    parser.add_argument(
        '--title', default=None, help='matplotlib figure title (default: None)'
    )
    parser.add_argument(
        '--xlabel', default='Timesteps', help='matplotlib figure xlabel'
    )
    parser.add_argument(
        '--ylabel', default='episode_reward', help='matplotlib figure ylabel'
    )

    parser.add_argument(
        '--ylabel2', default='normalized_episode_reward', help='matplotlib figure ylabel'
    )

    args = parser.parse_args()

    for algo in args.algos:
        path = os.path.join(args.root_dir, args.task, algo, 'test')
        result = convert_tfenvents_to_csv(path, args.xlabel, args.ylabel,args.ylabel2 )
        merge_csv(result, path, args.xlabel, args.ylabel, args.ylabel2)

    