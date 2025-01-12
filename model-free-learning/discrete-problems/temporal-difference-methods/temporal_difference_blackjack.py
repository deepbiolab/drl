import sys
import gymnasium as gym
import numpy as np
from collections import defaultdict
from plot_utils import plot_blackjack_values, plot_policy

def epsilon_greedy(actions, epsilon):
    nA = len(actions)
    probs = np.ones(nA) * (epsilon / nA)
    best_action = np.argmax(actions)
    probs[best_action] = 1 - epsilon + (epsilon / nA)
    return probs

def td_control_sarsa(env, num_episodes, alpha, gamma=1.0, eps_decay=0.999, eps_min=0.05):
    nA = env.action_space.n
    Q = defaultdict(lambda: np.zeros(nA))
    epsilon = 1.0
    
    for i_episode in range(1, num_episodes+1):
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
            
        epsilon = max(epsilon*eps_decay, eps_min)
        
        # observe S_0
        state, info = env.reset()
        
        # choose action A_0 using policy derived from Q table
        # S_0   --> A_0
        probs = epsilon_greedy(Q[state], epsilon)
        action = np.random.choice(np.arange(nA), p=probs)

        # in an episode
        while True:
            
            # take action A_t and observe R_t, S_t+1
            # A_t   --> R_t, S_t+1
            next_state, reward, done, truncated, info = env.step(action)
            
            # until next_state is terminal
            # reward R_T as final return
            if done:
                Q[state][action] = Q[state][action] + alpha * (reward - Q[state][action])
                break
            
            # choose action A_t+1 using same policy on next state from Q table
            # S_t+1 --> A_t+1 
            next_probs = epsilon_greedy(Q[next_state], epsilon)
            next_action = np.random.choice(np.arange(nA), p=next_probs)

            # update Q table
            G_estimate = reward + gamma * Q[next_state][next_action]
            Q[state][action] = Q[state][action] + alpha * (G_estimate - Q[state][action])

            # update state, action
            state, action = next_state, next_action
    
    # get policy
    policy = {k: np.argmax(v) for k, v in Q.items()}
    return policy, Q


def td_control_sarsamax(env, num_episodes, alpha, gamma=1.0, eps_decay=0.999, eps_min=0.05):
    nA = env.action_space.n
    Q = defaultdict(lambda: np.zeros(nA))
    epsilon = 1.0
    for i_episode in range(1, num_episodes+1):
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        epsilon = max(epsilon*eps_decay, eps_min)
        
        # observe S_0
        state, info = env.reset()
        
        # in an episode
        while True:
            # choose action A_t using policy derived from Q table
            # S_t --> A_t
            probs = epsilon_greedy(Q[state], epsilon)
            action = np.random.choice(np.arange(nA), p=probs)
            
            # take action A_t and observe R_t, S_t+1
            # A_t --> R_t, S_t+1
            next_state, reward, done, truncated, info = env.step(action)
            
            # until next_state is terminal
            # reward R_T as final return
            if done:
                Q[state][action] = Q[state][action] + alpha * (reward - Q[state][action])
                break
                
            # choose best action value in next state from Q table (off-policy)
            # S_t+1 --> max(A_t+1)
            next_action = np.argmax(Q[next_state])
            
            # update Q table using max action value
            G_estimate = reward + gamma * Q[next_state][next_action]
            Q[state][action] = Q[state][action] + alpha * (G_estimate - Q[state][action])
            
            # update state only (no need to update action in Q-learning)
            state = next_state
    
    # get policy
    policy = {k: np.argmax(v) for k, v in Q.items()}
    return policy, Q


def td_control_expected_sarsa(env, num_episodes, alpha, gamma=1.0, eps_decay=0.999, eps_min=0.05):
    nA = env.action_space.n
    Q = defaultdict(lambda: np.zeros(nA))
    epsilon = 1.0
    
    for i_episode in range(1, num_episodes+1):
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
            
        epsilon = max(epsilon*eps_decay, eps_min)
        
        # observe S_0
        state, info = env.reset()
        
        # in an episode
        while True:
            # choose action A_t using policy derived from Q table
            # S_t   --> A_t
            probs = epsilon_greedy(Q[state], epsilon)
            action = np.random.choice(np.arange(nA), p=probs)

            # take action A_t and observe R_t, S_t+1
            # A_t   --> R_t, S_t+1, 
            next_state, reward, done, truncated, info = env.step(action)

            # until next_state is terminal
            # reward R_T as final return
            if done:
                Q[state][action] = Q[state][action] + alpha * (reward - Q[state][action])
                break
            
            # calculate the expected action value for the next state
            # S_t+1 --> A_t+1 using same policy
            next_probs = epsilon_greedy(Q[next_state], epsilon)
            expection = np.dot(next_probs, Q[next_state])

            # update Q table
            G_estimate = reward + gamma * expection
            Q[state][action] = Q[state][action] + alpha * (G_estimate - Q[state][action])

            # update state only
            state = next_state

    # get policy
    policy = {k: np.argmax(v) for k, v in Q.items()}
    return policy, Q


def main():
    # Set random seed for reproducibility
    np.random.seed(0)
    
    # Create environment
    env = gym.make('Blackjack-v1')
    
    print("Environment Info:")
    print("Observation Space:", env.observation_space)
    print("Action Space:", env.action_space)
    print("\n")

    # Part 1: SARSA
    print("Part 1: SARSA (On-Policy TD Control)")
    print("Running SARSA...")
    policy_sarsa, Q_sarsa = td_control_sarsa(env, 500000, alpha=0.02)
    
    # Plot results
    V_sarsa = dict((k, np.max(v)) for k, v in Q_sarsa.items())
    plot_blackjack_values(V_sarsa)
    plot_policy(policy_sarsa)
    print("\nSARSA completed.\n")

    # Part 2: Q-Learning (SARSA-Max)
    print("Part 2: Q-Learning (Off-Policy TD Control)")
    print("Running Q-Learning...")
    policy_sarsamax, Q_sarsamax = td_control_sarsamax(env, 500000, alpha=0.02)
    
    # Plot results
    V_sarsamax = dict((k, np.max(v)) for k, v in Q_sarsamax.items())
    plot_blackjack_values(V_sarsamax)
    plot_policy(policy_sarsamax)
    print("\nQ-Learning completed.\n")

    # Part 3: Expected SARSA
    print("Part 3: Expected SARSA")
    print("Running Expected SARSA...")
    policy_expected, Q_expected = td_control_expected_sarsa(env, 500000, alpha=0.02)
    
    # Plot results
    V_expected = dict((k, np.max(v)) for k, v in Q_expected.items())
    plot_blackjack_values(V_expected)
    plot_policy(policy_expected)
    print("\nExpected SARSA completed.")

if __name__ == "__main__":
    main()