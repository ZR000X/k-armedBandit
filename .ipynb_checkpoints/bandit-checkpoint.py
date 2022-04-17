from math import exp, sqrt, pi, log
import matplotlib.pyplot as plt
import numpy as np
from time import time

def bp1(k, N, step, explore, var=1, dx=0.01, figures=True):
    """
    Simulates a k-armed Bandit Problem as descibred in Sutton, Barto (2020)
    
    Uses a constant variance
    """
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

    Qa = np.zeros(k)
    Q = [np.array([0]) for _ in range(k)]
    y = np.array([0])
    A = np.array([])
    for n in range(N):
        if np.random.rand() < explore:
            a = np.random.randint(k)            
        else:
            w = np.where(Qa == max(Qa))
            a = w[0][np.random.randint(len(w[0]))]
        A = np.append(A, a)
        r = bandit(a)
        Qa[a] += (r-Qa[a])*(step(n) if callable(step) else step)
        for i in range(k):
            Q[i] = np.append(Q[i], Qa[i])
        y = np.append(y, r) 

    if figures:
        x = np.arange(-k, k, dx)
        fig, ax = plt.subplots()
        ax.set_title('The k Bandits')
        plt.xlabel('Rewards')
        plt.yticks([])
        for i in range(k):
            ax.plot(x, apply(bandits[i], x), label=str(i))
        ax.legend()
        
        x = np.arange(0,N+1,1)
        fig, ax = plt.subplots()
        ax.set_title('Rewards')
        plt.xlabel('Timestep')
        plt.ylabel('Reward')
        plt.xticks([i for i in range(k)])   
        ax.scatter(x, y, s=0.1)

        fig, ax = plt.subplots()
        ax.set_title('Actions taken')
        plt.xlabel('Action')
        plt.ylabel('Frequency')
        plt.xticks([i for i in range(k)])        
        Ac = [len(np.where(A == i)[0]) for i in range(k)]
        plt.bar(np.arange(k), Ac)

        fig, ax = plt.subplots()
        ax.set_title('Action Reward Estimates')
        plt.xlabel('Timestep')
        plt.ylabel('Reward Estimate')
        for i in range(k):
            ax.plot(np.arange(N+1), Q[i], label=str(i))
        ax.legend()

    end = time()
    print("Runtime:", end-start)