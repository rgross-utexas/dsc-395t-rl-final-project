import argparse
from typing import Dict

from matplotlib import pyplot as plt
import numpy as np


class Arm:

    def __init__(self, mean, std, num_steps, initial_value = None):
        self.mean = mean
        self.std = std
        self.data = np.random.normal(mean, std, num_steps)
        if initial_value is not None:
            self.data[0] = initial_value

    def __str__(self):
        return f'{self.mean=}, {self.std=}, data_mean={np.average(self.data)}, data_std={np.std(self.data)}'


def main(args) -> None:

    print(f'{args=}')

    # get all the args
    num_arms = args.num_arms
    mean = args.mean
    std = args.std
    num_steps = args.num_steps
    num_runs = args.num_runs
    epsilons = args.epsilons
    initial_value = args.initial_value
    alpha = args.alpha
    output_filename = args.output_filename

    data = {}

    for epsilon in epsilons:

        # track all the averages as we go
        o_avg = np.zeros(num_steps, dtype=float)
        r_avg = np.zeros(num_steps, dtype=float)

        # save the averages for each epsilon
        data[epsilon] = {'r': r_avg, 'o': o_avg}

        for run in range(1, num_runs + 1):

            print(f"Starting {run=}...")

            # track this over each run
            if initial_value is not None:
                q_run = np.array([initial_value] * num_arms, dtype=float)
            else:
                q_run = np.zeros(num_arms, dtype=float)
            n_run = np.zeros(num_arms, dtype=int)

            # create the arms
            arm_means = np.random.normal(mean, std, num_arms)
            arms = []
            for arm_mean in arm_means:
                arm = Arm(arm_mean, std, num_steps, initial_value)
                arms.append(arm)

            # track the optimal action
            optimal_action = np.argmax(arm_means)

            for step in range(num_steps):

                # select the arm, possibly at random based on epsilon
                e = np.random.uniform(0, 1)
                if e > epsilon:
                    q_run_max = q_run.max()
                    choice = np.flatnonzero(q_run == q_run_max)
                    action = np.random.choice(choice)
                else:
                    action = np.random.randint(0, num_arms)

                # track optimal actions
                o_avg[step] = o_avg[step] + ((action == optimal_action) - o_avg[step]) / run

                # track rewards
                r = arms[action].data[step]
                r_avg[step] = r_avg[step] + (r - r_avg[step]) / run

                n_run[action] += 1
                q_run[action] = q_run[action] + alpha * (r - q_run[action])

    render_figure(data, output_filename)


def render_figure(data: Dict, output_filename: str):

    # render the figure and write it to file
    fig, axes = plt.subplots(2, 1)
    fig.set_figwidth(12)
    fig.set_figheight(12)

    axes[1].set_ylim([0., 1.])

    for epsilon, data in data.items():
        axes[0].plot(data['r'], label=f'epsilon={epsilon}')
        axes[1].plot(data['o'], label=f'epsilon={epsilon}')

    axes[0].legend(loc='lower right')
    axes[0].title.set_text("Average Reward Value")
    axes[1].legend(loc='lower right')
    axes[1].title.set_text("Optimal Action Ratio")

    plt.savefig(output_filename)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='k-armed-testbed')

    parser.add_argument('--num_arms', type=int, default=10)
    parser.add_argument('--mean', type=float, default=0.0)
    parser.add_argument('--std', type=float, default=1.0)
    parser.add_argument('--num_steps', type=int, default=1000)
    parser.add_argument('--num_runs', type=int, default=2000)
    parser.add_argument('--epsilons', nargs='*', default=[1.0, .1, .05, .01, 0.0])
    parser.add_argument('--initial_value', type=float, default=5)
    parser.add_argument('--alpha', type=float, default=.1)
    parser.add_argument('--output_filename', type=str, default='k-armed-testbed-non-stat-1000-500-5.png')

    main(args=parser.parse_args())
