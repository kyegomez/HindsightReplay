[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# Hindsight Experience Replay (HER)
=================================

Hindsight Experience Replay (HER) is a reinforcement learning technique that makes use of failed experiences to learn how to achieve goals. It does this by storing additional transitions in the replay buffer where the goal is replaced with the achieved state. This allows the agent to learn from a hindsight perspective, as if it had intended to reach the achieved state from the beginning.

## Implementation
--------------

This repository contains a Python implementation of HER using PyTorch. The main class is `HindsightExperienceReplay`, which represents a replay buffer that stores transitions and allows for sampling mini-batches of transitions.

The `HindsightExperienceReplay` class takes the following arguments:

-   `state_dim`: The dimension of the state space.
-   `action_dim`: The dimension of the action space.
-   `buffer_size`: The maximum size of the replay buffer.
-   `batch_size`: The size of the mini-batches to sample.
-   `goal_sampling_strategy`: A function that takes a tensor of goals and returns a tensor of goals. This function is used to dynamically sample goals for replay.

The `HindsightExperienceReplay` class has the following methods:

-   `store_transition(state, action, reward, next_state, done, goal)`: Stores a transition and an additional transition where the goal is replaced with the achieved state in the replay buffer.
-   `sample()`: Samples a mini-batch of transitions from the replay buffer and applies the goal sampling strategy to the goals.
-   `__len__()`: Returns the current size of the replay buffer.

## Usage
-----

Here is an example of how to use the `HindsightExperienceReplay` class:

```python
# Define a goal sampling strategy
def goal_sampling_strategy(goals):
    noise = torch.randn_like(goals) * 0.1
    return goals + noise

# Define the dimensions of the state and action spaces, the buffer size, and the batch size
state_dim = 10
action_dim = 2
buffer_size = 10000
batch_size = 64

# Create an instance of the HindsightExperienceReplay class
her = HindsightExperienceReplay(state_dim, action_dim, buffer_size, batch_size, goal_sampling_strategy)

# Store a transition
state = np.random.rand(state_dim)
action = np.random.rand(action_dim)
reward = np.random.rand()
next_state = np.random.rand(state_dim)
done = False
goal = np.random.rand(state_dim)
her.store_transition(state, action, reward, next_state, done, goal)

# Sample a mini-batch of transitions
sampled_transitions = her.sample()
if sampled_transitions is not None:
    states, actions, rewards, next_states, dones, goals = sampled_transitions
```


In this example, we first define a goal sampling strategy function and the dimensions of the state and action spaces, the buffer size, and the batch size. We then create an instance of the `HindsightExperienceReplay` class, store a transition, and sample a mini-batch of transitions. The states, actions, rewards, next states, done flags, and goals are returned as separate tensors.

## Customizing the Goal Sampling Strategy
--------------------------------------

The `HindsightExperienceReplay` class allows you to define your own goal sampling strategy by passing a function to the constructor. This function should take a tensor of goals and return a tensor of goals.

Here is an example of a goal sampling strategy function that adds random noise to the goals:

```
def goal_sampling_strategy(goals):
    noise = torch.randn_like(goals) * 0.1
    return goals + noise
```

In this example, the function adds Gaussian noise with a standard deviation of 0.1 to the goals. You can customize this function to implement any goal sampling strategy that suits your needs.

## Contributing
------------

Contributions to this project are welcome. If you find a bug or think of a feature that would be nice to have, please open an issue. If you want to contribute code, please fork the repository and submit a pull request.

## License
-------

This project is licensed under the MIT License. See the [LICENSE](https://domain.apac.ai/LICENSE) file for details.