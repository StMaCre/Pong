# Import 
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import random
from collections import deque
import matplotlib.pyplot as plt
import cv2


# Helper functions 
def preprocess(frame):
    # Convert the image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Crop the screen
    cropped_frame = gray[34:34 + 160, :160]
    
    # Normalize pixel values
    normalized_frame = cropped_frame / 255.0
    
    # Resize the frame to 84x84 pixels
    preprocessed_frame = cv2.resize(normalized_frame, (84, 84))
    
    return preprocessed_frame

# Architecture 
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(input_dim[0], 32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(512, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)
# Replay Buffer 
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def can_provide_sample(self, batch_size):
        return len(self.buffer) >= batch_size

# Hyperparameters
learning_rate = 0.0001
gamma = 0.97  
batch_size = 64  
memory_size = 10000  
target_update = 1000  
epsilon_start = 1.0
epsilon_end = 0.1
epsilon_decay = 10000  

# Initialize network and optimizer
input_dim = (4, 84, 84)
output_dim = env.action_space.n  
policy_net = DQN(input_dim, output_dim) 
target_net = DQN(input_dim, output_dim)  
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)

# Initialize variables
total_rewards = deque(maxlen=100) 
episode = 0
replay_memory = deque(maxlen=memory_size)
steps = 0

# Replay buffer
replay_buffer = ReplayBuffer(10000)


# Training loop
while steps <= 1000000:  
    state_deque = deque(maxlen=4)  
    initial_frame, _ = env.reset()
    for _ in range(4): 
        state_deque.append(preprocess(initial_frame))
    state = np.array(state_deque) 
    total_reward = 0
    done = False

    while not done:

        steps += 1  

        # Epsilon-greedy action selection
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-1. * steps / epsilon_decay)

        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = policy_net(state_tensor)
                action = q_values.argmax().item()

        next_state, reward, done, *_ = env.step(action)
        total_reward += reward

        next_frame  = preprocess(next_state)
        state_deque.append(next_frame)   #
        next_state_array = np.array(state_deque)


        # Add experience to replay buffer
        replay_buffer.push(state, action, reward, next_state_array, done)

        state = next_state_array  # Update the current state

        # Training from replay buffer
        if replay_buffer.can_provide_sample(batch_size):
            mini_batch = replay_buffer.sample(batch_size)
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*mini_batch)

            state_batch = torch.FloatTensor(np.array(state_batch))
            action_batch = torch.LongTensor(action_batch)
            reward_batch = torch.FloatTensor(reward_batch)
            next_state_batch = torch.FloatTensor(np.array(next_state_batch))
            done_batch = torch.BoolTensor(done_batch)

            current_q_values = policy_net(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze()
            next_q_values = target_net(next_state_batch).max(1)[0]
            target_q_values = reward_batch + (gamma * next_q_values * ~done_batch)

            loss = F.mse_loss(current_q_values, target_q_values)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if steps % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

    episode += 1  
    total_rewards.append(total_reward)
    moving_average = np.mean(total_rewards)
    print(f"Episode: {episode}, Total Reward: {total_reward}, Moving Average: {moving_average}")
    reward_list.append(moving_average) 

    # Print step updates
    if steps % 1000 == 0:
        print(f"Step: {steps}")

# Plotting the moving average 
plt.plot(reward_list)
plt.ylabel('Moving Average of Rewards')
plt.xlabel('Episodes')
plt.show()
