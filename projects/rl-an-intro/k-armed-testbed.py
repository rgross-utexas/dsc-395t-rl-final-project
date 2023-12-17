import argparse
from typing import Dict

from matplotlib import pyplot as plt
import numpy as np


class Arm:

    def __init__(self, mean, std, num_steps):
        self.mean = mean
        self.std = std
        self.data = np.random.normal(mean, std, num_steps)

    def __str__(self):
        return f'{self.mean=}, {self.std=}, data_mean={np.average(self.data)}, data_std={np.std(self.data)}'


def main(args) -> None:

    args = vars(args)
    print(f'{args=}')

    data = do_run(args)

    render_figure(data, args)


def do_run(args: Dict) -> Dict:

    data = {}

    num_arms = args.get('num_arms')
    mean = args.get('mean')
    std = args.get('std')
    num_steps = args.get('num_steps')
    num_runs = args.get('num_runs')
    epsilons = args.get('epsilons')
    initial_value = args.get('initial_value')
    alpha = args.get('alpha')
    run_type = args.get('run_type')
    explore_type = args.get('explore_type')
    ucb_consts = args.get('ucb_consts')

    if explore_type == 'ucb':
        values = ucb_consts
    else:
        values = epsilons

    for v in values:

        # track all the averages as we go
        o_avg = np.zeros(num_steps, dtype=float)
        r_avg = np.zeros(num_steps, dtype=float)

        # save the averages for each epsilon
        data[v] = {'r': r_avg, 'o': o_avg}

        for run in range(1, num_runs + 1):

            print(f"Starting {run=}...")

            # track this over each run
            q_run = np.array([initial_value] * num_arms, dtype=float)
            n_run = np.zeros(num_arms, dtype=int)

            # create the arms
            arm_means = np.random.normal(mean, std, num_arms)
            arms = []
            for arm_mean in arm_means:
                arm = Arm(arm_mean, std, num_steps)
                arms.append(arm)

            # track the optimal action
            optimal_action = np.argmax(arm_means)

            for step in range(num_steps):

                if explore_type == 'ucb':
                    action = np.argmax(q_run + v * np.sqrt(np.log(step + 1)/n_run))
                else:
                    # select the action, possibly at random based on epsilon
                    e = np.random.uniform(0, 1)
                    if e > v:
                        # noinspection PyArgumentList
                        q_run_max = q_run.max()
                        choice = np.flatnonzero(q_run == q_run_max)
                        action = np.random.choice(choice)
                    else:
                        action = np.random.randint(0, num_arms)

                # track which action was taken
                n_run[action] += 1

                # track optimal actions
                o_avg[step] = o_avg[step] + ((action == optimal_action) - o_avg[step]) / run

                # track rewards
                r = arms[action].data[step]
                r_avg[step] = r_avg[step] + (r - r_avg[step]) / run

                # choose which update rule here
                if run_type == 'stationary':
                    q_run[action] = q_run[action] + alpha * (r - q_run[action])
                else:
                    q_run[action] = q_run[action] + (r - q_run[action]) / n_run[action]

    return data


def render_figure(data: Dict, args: Dict):

    if args.get('explore_type') == 'ucb':
        label = 'UCB constant'
    else:
        label = 'epsilon'

    # render the figure and write it to file
    fig, axes = plt.subplots(2, 1)
    fig.set_figwidth(12)
    fig.set_figheight(12)

    axes[1].set_ylim([0., 1.])

    for epsilon, data in data.items():
        axes[0].plot(data['r'], label=f'{label}={epsilon}')
        axes[1].plot(data['o'], label=f'{label}={epsilon}')

    axes[0].legend(loc='lower right')
    axes[0].title.set_text("Average Reward Value")
    axes[1].legend(loc='lower right')
    axes[1].title.set_text("Optimal Action Ratio")

    plt.savefig(args.get('output_filename'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='k-armed-testbed')

    parser.add_argument('--num_arms', type=int, default=10)
    parser.add_argument('--mean', type=float, default=0.0)
    parser.add_argument('--std', type=float, default=1.0)
    parser.add_argument('--num_steps', type=int, default=1000)
    parser.add_argument('--num_runs', type=int, default=2000)
    parser.add_argument('--epsilons', nargs='*',
                        default=[.1, .05, .01, 0.0])
    parser.add_argument('--initial_value', type=float, default=0)
    parser.add_argument('--alpha', type=float, default=.1)
    parser.add_argument('--run_type',
                        choices=['default', 'stationary'],
                        default='default')
    parser.add_argument('--explore_type',
                        choices=['e_greedy', 'ucb'],
                        default='e_greedy')
    parser.add_argument('--ucb_consts', nargs='*', default=[1, 2, 5, 10])
    parser.add_argument('--output_filename', type=str,
                        default='images/k-armed-testbed-stat.png')

    main(args=parser.parse_args())
