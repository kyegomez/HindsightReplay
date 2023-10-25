import torch 
from hindsight import HindsightExperienceReplay
from numpy import np





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
her = HindsightExperienceReplay(
    state_dim, action_dim, buffer_size, batch_size, goal_sampling_strategy
)

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


