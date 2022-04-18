'''
-------------------------------------------------
    k-Armed Testbed (Reinforcement Learning: An Introduction, Sutton, Barto, 2020)
    Created by Shaun Schoeman 2022-04-17, last updated 2022-04-18
-------------------------------------------------
'''

from math import exp, sqrt, pi, log
import matplotlib.pyplot as plt
import numpy as np
from time import time

def bp1(k, N, step, explore, var=1, dx=0.01, figures={"bandits","rewards","actions","actions","estimates"}):
    """
    Simulates a k-armed Bandit Problem as descibred in Sutton, Barto (2020)
    
    @param k: the number of bandits to be chosen from
    @param N: the timesteps used in the simulation
    @param step: the stepsize used in updated action-values
    @param explore: the probability given to 
    @param var: uses a constant variance  
    """
    explore = round(explore,2)

    def arm(m, s):
        return lambda x: 1/sqrt(2*pi*s**2)*exp(-1/2*((x-m)/s)**2)

    def apply(func, array):
        return np.array([func(i) for i in array])

    start = time()

    means = [np.random.rand()*i for i in range(k)]
    bandits = [arm(means[i], var) for i in range(k)]

    def bandit(i):
        sign = 1
        if np.random.rand() < 0.5:
            sign *= -1
        x = bandits[i](np.random.rand()*sign+means[i])
        return sqrt(-2*log(sqrt(2*pi*var**2)*x))+means[i]

    act_val_ests = np.zeros(k)
    act_val_ests_history = [np.array([0]) for _ in range(k)]
    rewards = np.array([0])
    actions = np.array([])
    for n in range(N):
        if np.random.rand() < explore:
            a = np.random.randint(k)            
        else:
            w = np.where(act_val_ests == max(act_val_ests))
            a = w[0][np.random.randint(len(w[0]))]
        actions = np.append(actions, a)
        r = bandit(a)
        act_val_ests[a] += (r-act_val_ests[a])*(step(n) if callable(step) else step)
        for i in range(k):
            act_val_ests_history[i] = np.append(act_val_ests_history[i], act_val_ests[i])
        rewards = np.append(rewards, r) 

    if not figures is None:
        if "bandits" in figures:
            x = np.arange(-k, k, dx)
            fig, ax = plt.subplots()
            ax.set_title(f'The k Bandits, e={explore}')
            plt.xlabel('Rewards')
            plt.yticks([])
            for i in range(k):
                ax.plot(x, apply(bandits[i], x), label=str(i))
            ax.legend()        
        if "rewards" in figures:
            x = np.arange(0,N+1,1)
            fig, ax = plt.subplots()
            ax.set_title(f'Rewards, e={explore}')
            plt.xlabel('Timestep')
            plt.ylabel('Reward')
            plt.xticks([i for i in range(k)])   
            ax.scatter(x, rewards, s=0.1)
        if "actions" in figures:
            fig, ax = plt.subplots()
            ax.set_title(f'Actions taken, e={explore}')
            plt.xlabel('Action')
            plt.ylabel('Frequency')
            plt.xticks([i for i in range(k)])        
            Ac = [len(np.where(actions == i)[0]) for i in range(k)]
            plt.bar(np.arange(k), Ac)
        if "estimates" in figures:
            fig, ax = plt.subplots()
            ax.set_title(f'Action Reward Estimates, e={explore}')
            plt.xlabel('Timestep')
            plt.ylabel('Reward Estimate')
            for i in range(k):
                ax.plot(np.arange(N+1), act_val_ests_history[i], label=str(i))
            ax.legend()

    end = time()
    return {
        "runtime": end-start,
        "means": means,
        "bandits": bandits,
        "action_values": act_val_ests_history,
        "actions": actions,
        "rewards": rewards
    }

def compare_exploration(k, N, step, erange=np.arange(0,1.1,0.1), var=1, dx=0.01, figures=None, figure=True):
    data = [bp1(k=k, N=N, step=step, explore=e, var=var, dx=dx, figures=figures)["rewards"] for e in erange]
    plot_data = list()
    fig, ax = plt.subplots()
    ax.set_title('Reward Accumulation per Exploration')
    plt.xlabel('Timestep')
    plt.ylabel('Reward Estimate')
    x = np.arange(N+1)
    for e in range(len(erange)):
        plot_data.append(np.array([0]))
        for f in range(N):
            plot_data[e] = np.append(plot_data[e], plot_data[e][-1]+data[e][f])
        ax.plot(x, plot_data[e], label=str(round(erange[e],2)))
    ax.legend()
    return {
        "erange": erange,
        "plot_data": plot_data
    }

