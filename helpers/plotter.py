# Borrow a lot from tianshou:
# https://github.com/thu-ml/tianshou/blob/master/examples/mujoco/plotter.py
import csv
import os
import re

import pickle
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import tqdm
import argparse
import wandb
from tensorboard.backend.event_processing import event_accumulator

COLORS = (
    [
        # deepmind style
        '#0072B2',
        '#009E73',
        '#D55E00',
        '#CC79A7',
        # '#F0E442',
        '#d73027',  # RED
        # built-in color
        'blue',
        'red',
        'pink',
        'cyan',
        'magenta',
        'yellow',
        'black',
        'purple',
        'brown',
        'orange',
        'teal',
        'lightblue',
        'lime',
        'lavender',
        'turquoise',
        'darkgreen',
        'tan',
        'salmon',
        'gold',
        'darkred',
        'darkblue',
        'green',
        # personal color
        '#313695',  # DARK BLUE
        '#74add1',  # LIGHT BLUE
        '#f46d43',  # ORANGE
        '#4daf4a',  # GREEN
        '#984ea3',  # PURPLE
        '#f781bf',  # PINK
        '#ffc832',  # YELLOW
        '#000000',  # BLACK
    ]
)


def convert_tfenvents_to_csv(root_dir, xlabel, ylabel):
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
            content = [[xlabel, ylabel]]
            for test_rew in ea.scalars.Items('eval/'+ylabel):
                content.append(
                    [
                        round(test_rew.step, 4),
                        round(test_rew.value, 4),
                    ]
                )
            csv.writer(open(output_file, 'w')).writerows(content)
            result[output_file] = content
    return result


def merge_csv(csv_files, root_dir, xlabel, ylabel):
    """Merge result in csv_files into a single csv file."""
    assert len(csv_files) > 0
    sorted_keys = sorted(csv_files.keys())
    sorted_values = [csv_files[k][1:] for k in sorted_keys]
    content = [
        [xlabel, ylabel+'_mean', ylabel+'_std']
    ]
    for rows in zip(*sorted_values):
        array = np.array(rows)
        assert len(set(array[:, 0])) == 1, (set(array[:, 0]), array[:, 0])
        line = [rows[0][0], round(array[:, 1].mean(), 4), round(array[:, 1].std(), 4)]
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


def plot_histogram(data, y_label,):
    color = [f'tab:{COLORS[0]}', f'tab:{COLORS[1]}', f'tab:{COLORS[2]}']
    """
    Plot histograms of data[1][y_label], data[2][y_label], data[3][y_label],
    each in its own color.
    
    Parameters
    ----------
    data : dict of DataFrame-like
        data[i][y_label] should be iterable of values.
    y_label : str
        Column/key to plot.
    """
    plt.figure(figsize=(8, 5))
    
    # Loop over the three iterations
    for i, color in zip(np.arange(4), COLORS):
        plt.hist(
            data[i][y_label],
            bins=50,
            density=True,
            alpha=0.6,        # semi‐transparent so overlaps show
            color=color,
            label=f'Iteration {i}' if i != 0 else 'Ground Truth',
        )
    
    plt.xlabel(y_label)
    plt.ylabel('Count')
    plt.title(f'Histogram of {y_label} over {i} iterations')
    plt.legend()
    plt.tight_layout()
    out_file = os.path.join(args.output_path, f'{y_label}.png')
    plt.savefig(out_file, dpi=args.dpi, bbox_inches='tight')
    plt.show()


def plot_policy(eval_env, state, all_states, title, legend=None):
    """

    Plot the policy for the given state and environment.
    Args: 
        eval_env: The evaluation environment.
        state ([max_steps, forecast_horizon*num_features]): The predicted state to plot. Includes the first p-level.
        all_states ([max_steps+1, forecast_horizon*num_features]): The real states including the first inputted state.
        writer: The writer to log the plot.
    """

    input_color = 'tab:blue'
    pred_color = 'tab:red' #label="input",
    gt_color = 'tab:red'
    rl_color = 'tab:blue'
    hr_color = 'tab:orange'
    pulsat_color = 'tab:green'


    max_steps = eval_env.max_steps
    forecast_n = eval_env.world_model.forecast_horizon
    action_unnorm  = np.repeat(eval_env.episode_actions,forecast_n)


    state_unnorm = eval_env.world_model.unnorm_output(np.array(state).reshape(max_steps, forecast_n, -1))
    all_state_unnorm = eval_env.world_model.unnorm_output(np.array(all_states).reshape(max_steps+1, forecast_n, -1))
    first_action_unnorm = state_unnorm[0,:,-1] #normalized
    

    fig, ax1 = plt.subplots(figsize=(5, 5.5), dpi=300,  layout='constrained')  # Smaller plot size
                                    
    default_x_ticks = range(0, 181, 18)
    x_ticks = np.array(list(range(0, 31, 3)))
    plt.xticks(default_x_ticks, x_ticks)
    x1 = len(all_state_unnorm[0, :, 0].reshape(-1,1))
    x2 = len(all_state_unnorm[1:, :, 0].reshape(-1,1))
    ax1.axvline(x=x1, linestyle='--', c='black', alpha =0.7)


    
    line_obs, = ax1.plot(range(0, x1+x2), all_state_unnorm[:, :, 0].reshape(-1,1),  '--', label ='Observed MAP', alpha=0.5,  color=gt_color, linewidth=2.0)
    line_obs2, = ax1.plot(range(0, x1+x2), all_state_unnorm[:, :, 9].reshape(-1,1),  '--', label ='Observed HR', alpha=0.5,  color=hr_color, linewidth=2.0)
    line_obs3, = ax1.plot(range(0, x1+x2), all_state_unnorm[:, :, 7].reshape(-1,1),  '--', label ='Observed pulsat', alpha=0.5,  color=pulsat_color, linewidth=2.0)
    line_pred2, = ax1.plot(range(x1, x1+x2), state_unnorm[:, :, 9].reshape(-1,1),  label ='Predicted HR', alpha=0.5,  color=hr_color, linewidth=2.0)
    line_pred3, = ax1.plot(range(x1, x1+x2), state_unnorm[:, :, 7].reshape(-1,1), label ='Predicted PULSAT', alpha=0.5,  color=pulsat_color, linewidth=2.0)
    line_pred1, = ax1.plot(range(x1, x1+x2), state_unnorm[:, :, 0].reshape(-1,1), label ='Predicted MAP',color=pred_color,  linewidth=3)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    line_pl1, = ax2.plot(range(0, x1+x2),  all_state_unnorm[:,:,-1].reshape(-1), '--', label ='Input PL', alpha= 0.5, color=input_color, linewidth=2.0)
    line_pl2, = ax2.plot(range(x1, x1+x2), action_unnorm.reshape(-1,1),label ='Recommended PL', color=rl_color, linewidth=3)

    # Combined legend for all lines
    lines = [line_obs,line_obs2, line_obs3, line_pred2, line_pred3, line_pred1, line_pl1, line_pl2]
    labels = ['Observed MAP','Observed HR', 'Observed PULSAT', 'Predicted HR', 'Predicted PULSAT', 'Predicted MAP', 'Input PL', 'Recommended PL']
    if legend:
        ax1.legend(lines, labels, loc='upper right', bbox_to_anchor=(1.0, 1.37), fancybox=True, ncol=3, fontsize='medium')  # Legend at the bottom
    # ax1.set_ylabel('MAP (mmHg)', size="large", color='tab:red')
    ax1.tick_params(axis='y', colors='tab:red', labelsize='xx-large')
    ax1.tick_params(axis='x', labelsize='xx-large')
    ax1.set_xlabel('Time (hour)', fontsize=28)
    # ax1.set_title(f"{title}", size="xx-large", fontweight="bold")
    # ax2.set_ylabel('P-level', size="x-large", color='tab:blue', labelpad=10)
    ax2.tick_params(axis='y', colors='tab:blue', labelsize='xx-large')
    ax2.set_ylim(1, 10)
    ax1.set_ylim(10,130)
    # ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    # ax1.spines['left'].set_visible(False)
    # fig.subplots_adjust(left=0.14, right=0.88, top=0.90, bottom=0.24)
    # ax1.grid()
    wandb.log({f"eval_sample_{title}": wandb.Image(fig)})
    plt.close(fig)

    # ax1.set_ylabel('MAP (mmHg)', size="x-large", color='tab:red')

def plot_score_histograms(acp_list, ws_list, rwd_list, title):
    col_w_in   = 3.25   # one-column width (IEEE ~3.45", NeurIPS 3.25")
    row_h_in   = 1.8    # height per subplot (increase to make each panel taller)
    v_gap_in   = 0.8   # vertical gap between rows (inches)

    fig_h_in = 3 * row_h_in + 2 * v_gap_in

    fig, axes = plt.subplots(
        nrows=3, ncols=1,
        figsize=(col_w_in, fig_h_in),
        dpi=300,
        constrained_layout=False
    )
    # Control margins + gaps explicitly
    fig.subplots_adjust(left=0.18, right=0.92, top=0.92, bottom=0.12,
                    hspace=v_gap_in / row_h_in)

    

    configs = [
        ("ACP values",    acp_list, (0, 5)),
        ("WS values",     ws_list,  (-0.5, 1)),
        ("Reward values", rwd_list, (-12, 4)),
    ]

    for ax, (xlabel, vals, xlim) in zip(axes, configs):
        ax.hist(vals, edgecolor='black', linewidth=0.4)
        ax.set_xlim(*xlim)
        ax.set_xlabel(xlabel, fontsize=16)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=14)

    # # Single y-label for all
    # try:
    #     fig.supylabel("Episode Count", fontsize=10)
    # except AttributeError:
    #     for ax in axes: ax.set_ylabel("Episode Count", fontsize=28)

    # fig.suptitle(f"Distributions for {title.upper()}", fontsize=12, fontweight="bold", y=0.98)

    # Log once to W&B
    import wandb
    wandb.log({f"eval_distributions_{title}": wandb.Image(fig)})
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='plotter')
    parser.add_argument(
        '--root-dir', 
        #default='log/hopper-medium-replay-v0/mopo',
         default='log', help='root dir'
    )
    parser.add_argument(
        '--task', default='abiomed_plot', help='task'
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
        '--hist_ylabel', default='actions', help='matplotlib figure ylabel'
    )
    parser.add_argument(
        '--hist_ylabel2', default='rewards', help='matplotlib figure ylabel'
    )
    parser.add_argument(
        '--smooth', type=int, default=10, help='smooth radius of y axis (default: 0)'
    )
    parser.add_argument(
        '--colors', default=None, help='colors for different algorithms'
    )
    parser.add_argument('--show', action='store_true', help='show figure')
    parser.add_argument(
        '--output-path', type=str, help='figure save path', default="./figure.png"
    )
    parser.add_argument(
        '--dpi', type=int, default=200, help='figure dpi (default: 200)'
    )
    args = parser.parse_args()

    # args.task = 'halfcheetah-expert-v2'
    for algo in args.algos:
        path = os.path.join(args.root_dir, args.task, algo)
        result = convert_tfenvents_to_csv(path, args.xlabel, args.ylabel)
        merge_csv(result, path, args.xlabel, args.ylabel)

    # plt.style.use('seaborn')
    plot_figure(root_dir=args.root_dir, task=args.task, algo_list=args.algos, x_label=args.xlabel, y_label=args.ylabel, title=args.title, smooth_radius=args.smooth, color_list=args.colors)
    if args.output_path:
        plt.savefig(args.output_path, dpi=args.dpi, bbox_inches='tight')
    if args.show:
        plt.show()

    data = {}
    for i in [0,1]:
        test_path = os.path.join(args.data_path, f"dataset_test_{i}.pkl")
        with open(test_path, 'rb') as f:
            data[i] = pickle.load(f)
    plot_histogram(data, args.hist_ylabel)
    
    plot_histogram(data, args.hist_ylabel2)