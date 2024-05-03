import matplotlib.pyplot as plt
import numpy as np

GOAL = 100
STATES = np.arange(GOAL + 1)
HEAD_PROB = 0.4

def main():
    values = np.zeros(GOAL + 1)
    values[GOAL] = 1.0

    history = []

    while True:
        old_values = values.copy()
        history.append(old_values)

        for state in STATES[1:GOAL]:
            actions = np.arange(min(state, GOAL - state) + 1)
            returns = []
            for action in actions:
                returns.append(
                    HEAD_PROB * values[state + action] + (1 - HEAD_PROB) * values[state - action])
            new_value = np.max(returns)
            values[state] = new_value
        delta = abs(values - old_values).max()
        if delta < 1e-9:
            history.append(values)
            break

    policy = np.zeros(GOAL + 1)
    for state in STATES[1:GOAL]:
        actions = np.arange(min(state, GOAL - state) + 1)
        returns = []
        for action in actions:
            returns.append(
                HEAD_PROB * values[state + action] + (1 - HEAD_PROB) * values[state - action])

        policy[state] = actions[np.argmax(np.round(returns[1:], 5)) + 1]

    plt.figure(figsize=(10, 20))

    plt.subplot(2, 1, 1)
    for sweep, values in enumerate(history):
        plt.plot(values, label='sweep {}'.format(sweep))
    plt.xlabel('Capital')
    plt.ylabel('Value estimates')
    plt.legend(loc='best')

    plt.subplot(2, 1, 2)
    plt.scatter(STATES, policy)
    plt.xlabel('Capital')
    plt.ylabel('Final policy (stake)')

    plt.show()

if __name__ == '__main__':
    main()
