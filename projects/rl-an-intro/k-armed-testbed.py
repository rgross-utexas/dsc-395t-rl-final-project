import argparse
import random

import numpy as np


class Arm:

    def __init__(self, mean, std, num_steps):
        self.mean = mean
        self.std = std
        self.data = np.random.normal(mean, std, num_steps)

    def __str__(self):
        return f'{self.mean=}, {self.std=}, data_mean={np.average(self.data)}, data_std={np.std(self.data)}'

def main(args) -> None:

    print(f'{args=}')

    num_arms = args.num_arms
    mean = args.mean
    std = args.std
    num_steps = args.num_steps
    num_runs = 2000
    epsilon = args.epsilon

    # track all the averages as we go
    o_avg = np.zeros(num_steps, dtype=float)
    r_avg = np.zeros(num_steps, dtype=float)

    # track this over each run
    q_run = np.zeros(num_runs, dtype=float)
    n_run = np.zeros(num_runs, dtype=int)

    for run in range(1, num_runs + 1):

        arm_means = np.random.normal(mean, std, num_arms)
        print(f'{arm_means=}')
        true_best_arm = np.argmax(arm_means)
        print(f'{true_best_arm=}')

        # create the arms
        arms = []
        for arm_mean in arm_means:
            arm = Arm(arm_mean, std, num_steps)
            arms.append(arm)

        # randomly select the starting optimal action
        optimal_action = random.randint(0, num_arms - 1)

        # track the action history, in case we want to gather info from it later
        actions_taken = np.zeros(num_steps)
        optimal_action_taken = np.zeros(num_steps)
        random_action_taken = np.zeros(num_steps)

        rewards = np.zeros(num_steps)
        reward_averages = np.zeros(num_arms, dtype=float)

        # track the number of pulls, by arm
        num_pulls = np.zeros(num_arms, dtype=int)
        run_averages = np.zeros(num_steps)

        q_run.fill(0)
        n_run.fill(0)

        for step in range(num_steps):

            # select the arm, possibly at random
            if random.uniform(0, 1) < epsilon:
                action = np.random.choice(np.flatnonzero(q_run == q_run.max()))
                random_action_taken[step] = True
            else:
                action = optimal_action

            actions_taken[step] = action
            arm = arms[action]

            is_optimal = action == optimal_action
            optimal_action_taken[step] = is_optimal
            o_avg[step] = o_avg[step] + (is_optimal - o_avg[step]) / run

            # get the reward from the arm
            r = arm.data[step]
            r_avg[step] = r_avg[step] + (r - r_avg[step]) / run

            n_run[action] += 1
            q_run[action] = q_run[action] + (r - q_run[action]) / n_run[action]

            rewards[step] = r

            # calculate the average and the number of pulls for this arm
            reward_averages[action] = (num_pulls[action] * reward_averages[action] + r)/(num_pulls[action] + 1)


            num_pulls[action] += 1
            # print(f'{averages=}')

            # calculate the total average
            if step == 0:
                run_average = r
            else:
                run_average = (step * run_averages[step - 1] + r) / (step + 1)

            # print(f'{run_average=}')

            # track all the run averages
            run_averages[step] = run_average

            # pick the best arm based on the averages
            optimal_action = np.argmax(averages)
            # print(f'{best_arm_idx=}')

        print(f'{optimal_action=}')
        print(f'{averages=}')
        print(f'{num_pulls=}')
        print(f'{run_averages[-1]=}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='k-armed-testbed')
    parser.add_argument('--num_arms', type=int, default=10)
    parser.add_argument('--mean', type=float, default=0.0)
    parser.add_argument('--std', type=float, default=1.0)
    parser.add_argument('--num_steps', type=int, default=1000)
    parser.add_argument('--num_runs', type=int, default=2000)
    parser.add_argument('--epsilon', type=float, default=0.05)

    main(args=parser.parse_args())
