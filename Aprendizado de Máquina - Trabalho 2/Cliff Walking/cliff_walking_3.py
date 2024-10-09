import numpy as np
import matplotlib.pyplot as plt


def state_to_index(state, grid_size=(4,12)):
    return state[0] * grid_size[1] + state[1]

# Reward function
def reward_function(state, cliff_states):
    """
    Function to reward during the movements
    """
    if state in cliff_states:
        return -100  # Cliff penalty
    return -1  # Normal step cost

# Transition function: move based on action
def step(state, action, actions, cliff_states, start_state, goal_state, grid_size = (4, 12)):
    """
    Function to move inside the grid
    """
    if action == actions['up']:
        s_prime = (max(state[0] - 1, 0), state[1])
    elif action == actions['down']:
        s_prime = (min(state[0] + 1, grid_size[0] - 1), state[1])
    elif action == actions['left']:
        s_prime = (state[0], max(state[1] - 1, 0))
    else:  # action == actions['right']
        s_prime = (state[0], min(state[1] + 1, grid_size[1] - 1))

    if s_prime in cliff_states:
        return start_state, -100

    elif s_prime == goal_state:
        return s_prime, 0
    else:
        return s_prime, -1

# Epsilon-greedy action selection
def epsilon_greedy(Q, state, epsilon, actions):
    """
    Function to the next action with epsilon for randomness
    """
    if np.random.rand() < epsilon:
        return np.random.choice(list(actions.values()))  # Explore random action
    return np.argmax(Q[state_to_index(state)])  # Exploit best known action

def q_learning(Q, epsilon, alpha, gamma, num_episodes, start_state, goal_state, cliff_states, actions, grid_size=(4,12)):
    """
    Function to do q learning training loop
    """
    rewards_ep = []
    for _ in range(num_episodes):
        s = start_state
        total_rewards = 0
        
        while True:
            a = epsilon_greedy(Q, s, epsilon, actions=actions)
            s_prime, reward = step(
                state=s, 
                action=a, 
                actions=actions, 
                cliff_states=cliff_states, 
                start_state=start_state,
                goal_state=goal_state,
                grid_size=grid_size
                )
            
            best_next_action = np.argmax(Q[state_to_index(s_prime)])
            Q[state_to_index(s), a] = Q[state_to_index(s), a] + \
                alpha * (reward + gamma * Q[state_to_index(s_prime), best_next_action] - Q[state_to_index(s), a])
            
            s = s_prime
            total_rewards += reward
            if s == goal_state:
                break

        rewards_ep.append(total_rewards)
    return Q, rewards_ep

def plot_rewards(rewards_ep):
    """
    Function to plot rewards over iterations
    """
    plt.plot(rewards_ep)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Q-learning Performance on Cliff Walking')
    plt.show()

# Display the optimal Q-values for each state in a graphical window
def plot_grid(Q, cliff_states, start_state, goal_state, grid_size = (4,12)):
    """
    Function to plot the Q grid
    """
    optimal_q_grid = np.zeros(grid_size)

    for state in np.ndindex(grid_size):
        if state in cliff_states:
            optimal_q_grid[state[0], state[1]] = -100  # Mark cliff as -100
        elif state == goal_state:
            optimal_q_grid[state[0], state[1]] = 0 # Goal state as 0
        else:
            # Display the max Q-value for each state
            optimal_q_grid[state[0], state[1]] = np.max(Q[state_to_index(state)])

    # Masking the cliff and goal states for better visualization
    masked_q_grid = np.ma.masked_where(optimal_q_grid == -100, optimal_q_grid)

    # Plot the Q-values as a heatmap with the actual values inside each cell
    plt.figure(figsize=(10, 5))
    plt.imshow(masked_q_grid, cmap='viridis')
    plt.colorbar(label='Q-value')
    plt.title('Optimal Q-values Heatmap with Values Displayed')

    # Annotate each cell with its Q-value
    for state in np.ndindex(grid_size):
        if optimal_q_grid[state[0], state[1]] != -100:  # Skip the cliff cells
            plt.text(state[1], state[0], f'{optimal_q_grid[state[0], state[1]]:.2f}', ha='center', va='center', color='white', fontsize=8)
    plt.show()

def main():
    grid_height = 4
    grid_width = 12
    grid_size = (grid_height, grid_width)
    num_episodes = 500
    alpha = 0.1  # learning ratez
    gamma = 0.9  # discount factor
    epsilon = 0.1  # exploration rate
    start_state = (3, 0)
    goal_state = (3, 11)
    cliff_states = [(3, i) for i in range(1, 11)] 
    Q = np.zeros((grid_height * grid_width, 4))  


    actions = {'up': 0, 'down': 1, 'left': 2, 'right': 3}
    Q, rewards = q_learning(
        Q=Q, 
        epsilon=epsilon,
        alpha=alpha,
        gamma=gamma,
        num_episodes=num_episodes,
        start_state=start_state,
        goal_state=goal_state,
        cliff_states=cliff_states,
        actions=actions
        )
    
    plot_rewards(rewards)
    
    plot_grid(
        Q=Q, 
        cliff_states=cliff_states, 
        start_state=start_state,
        goal_state=goal_state,
        grid_size=grid_size
        )


if __name__ == "__main__":
    main()
