import torch 
import random
import numpy as np
from collections import deque

def goal_sampling_strategy(goals):
    noise = torch.rand_like(goals) * 0.1
    return goals + noise

class HindsightExperienceReplay:
    def __init__(self, state_dim, action_dim, buffer_size, batch_size, goal_sampling_strategy):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = deque(maxlen=buffer_size)
        self.goal_sampling_strategy = goal_sampling_strategy
    
    def store_transition(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        goal
    ):
        transition = (state, action, reward, next_state, done, goal)
        self.buffer.append(transition)

        # Store additional transition where the goal is replaced with the achievd state
        achieved_goal = next_state
        transition = (state, action, reward, next_state, done, achieved_goal)
        self.buffer.append(transition)

    def sample(self):
        mini_batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones, goals = zip(*mini_batch)


        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(np.float32(dones)).unsqueeze(1)
        goals = torch.FloatTensor(goals)

        # Apply goal sampling strategy
        goals = self.goal_sampling_strategy(states, goals)

        return states, actions, rewards, next_states, dones, goals
    
    def __len__(self):
        return len(self.buffer)
    