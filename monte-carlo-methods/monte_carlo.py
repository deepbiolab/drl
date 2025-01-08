import sys
import gymnasium as gym
import numpy as np
from collections import defaultdict
from plot_utils import plot_blackjack_values, plot_policy

def generate_episode_from_limit_stochastic(bj_env):
    episode = []
    state, info = bj_env.reset()
    while True:
        probs = [0.8, 0.2] if state[0] > 18 else [0.2, 0.8]
        action = np.random.choice(np.arange(2), p=probs)
        next_state, reward, done, truncated, info = bj_env.step(action)
        episode.append((state, action, reward))
        state = next_state
        if done:
            break
    return episode

def mc_prediction_q(env, num_episodes, generate_episode, gamma=1.0):
    returns_sum = defaultdict(lambda: np.zeros(env.action_space.n))
    N = defaultdict(lambda: np.zeros(env.action_space.n))
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    for i_episode in range(1, num_episodes+1):
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
            
        episode = generate_episode(env)
        states, actions, rewards = zip(*episode)
        discouts = np.array([gamma**i for i in range(len(rewards)+1)])
        
        visited = set()
        for i, state in enumerate(states):
            sa_pair = (state, actions[i])
            if sa_pair not in visited:
                visited.add(sa_pair)
                returns_sum[state][actions[i]] += sum(rewards[i:] * discouts[:-(i+1)])
                N[state][actions[i]] += 1
                Q[state][actions[i]] = returns_sum[state][actions[i]] / N[state][actions[i]]
    
    return Q

def epsilon_greedy(actions, epsilon):
    nA = len(actions)
    probs = np.ones(nA) * (epsilon / nA)
    best_action = np.argmax(actions)
    probs[best_action] = 1 - epsilon + (epsilon / nA)
    return probs

def mc_control_incremental_mean(env, num_episodes, eps_decay=0.999, eps_min=0.05, gamma=1.0):
    nA = env.action_space.n
    Q = defaultdict(lambda: np.zeros(nA))
    N = defaultdict(lambda: np.zeros(nA))
    epsilon = 1.0
    
    for i_episode in range(1, num_episodes+1):
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
            
        epsilon = max(epsilon*eps_decay, eps_min)
        
        episode = []
        state, info = env.reset()
        while True:
            probs = epsilon_greedy(Q[state], epsilon)
            action = np.random.choice(np.arange(nA), p=probs) if state in Q else env.action_space.sample()
            next_state, reward, done, truncated, info = env.step(action)
            episode.append((state, action, reward))
            state = next_state
            if done:
                break
                
        states, actions, rewards = zip(*episode)
        discount = np.array([gamma**i for i in range(len(states)+1)])
        
        visited = set()
        for i, state in enumerate(states):
            sa_pair = (state, actions[i])
            if sa_pair not in visited:
                visited.add(sa_pair)
                N[state][actions[i]] += 1
                G = sum(rewards[i:] * discount[:-(i+1)])
                Q[state][actions[i]] += (1/N[state][actions[i]]) * (G - Q[state][actions[i]])
    
    policy = {k: np.argmax(v) for k, v in Q.items()}
    return policy, Q

def mc_control(env, num_episodes, alpha, gamma=1.0, eps_decay=0.999, eps_min=0.05):
    nA = env.action_space.n
    Q = defaultdict(lambda: np.zeros(nA))
    epsilon = 1.0
    
    for i_episode in range(1, num_episodes+1):
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
            
        epsilon = max(epsilon*eps_decay, eps_min)
        
        episode = []
        state, info = env.reset()
        while True:
            probs = epsilon_greedy(Q[state], epsilon)
            action = np.random.choice(np.arange(nA), p=probs) if state in Q else env.action_space.sample()
            next_state, reward, done, truncated, info = env.step(action)
            episode.append((state, action, reward))
            state = next_state
            if done:
                break
                
        states, actions, rewards = zip(*episode)
        discount = np.array([gamma**i for i in range(len(states)+1)])
        
        visited = set()
        for i, state in enumerate(states):
            sa_pair = (state, actions[i])
            if sa_pair not in visited:
                visited.add(sa_pair)
                G = sum(rewards[i:] * discount[:-(i+1)])
                Q[state][actions[i]] = alpha * G + (1-alpha) * Q[state][actions[i]]
    
    policy = {k: np.argmax(v) for k, v in Q.items()}
    return policy, Q

def main():
    # Create environment
    env = gym.make('Blackjack-v1')
    
    print("Environment Info:")
    print("Observation Space:", env.observation_space)
    print("Action Space:", env.action_space)
    print("\n")

    # Part 1: MC Prediction
    print("Part 1: MC Prediction")
    print("Running MC prediction...")
    Q = mc_prediction_q(env, 500000, generate_episode_from_limit_stochastic)
    
    # Calculate and plot state-value function
    V_to_plot = dict((k, (k[0] > 18) * (np.dot([0.8, 0.2], v)) + 
                        (k[0] <= 18) * (np.dot([0.2, 0.8], v)))
                     for k, v in Q.items())
    plot_blackjack_values(V_to_plot)
    print("MC Prediction completed.\n")

    # Part 2: MC Control with Incremental Mean
    print("Part 2: MC Control with Incremental Mean")
    print("Running MC control with incremental mean...")
    policy_incremental, Q_incremental = mc_control_incremental_mean(env, 500000)
    
    # Plot results
    V_incremental = dict((k, np.max(v)) for k, v in Q_incremental.items())
    plot_blackjack_values(V_incremental)
    plot_policy(policy_incremental)
    print("\nMC Control with Incremental Mean completed.\n")

    # Part 3: MC Control with Constant-alpha
    print("Part 3: MC Control with Constant-alpha")
    print("Running MC control with constant-alpha...")
    policy_alpha, Q_alpha = mc_control(env, 500000, alpha=0.02)
    
    # Plot results
    V_alpha = dict((k, np.max(v)) for k, v in Q_alpha.items())
    plot_blackjack_values(V_alpha)
    plot_policy(policy_alpha)
    print("\nMC Control with Constant-alpha completed.")

if __name__ == "__main__":
    main()