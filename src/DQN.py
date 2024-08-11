from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import random

class DQN(nn.Module):
    def __init__(self, input_dim, hidden_layer_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_layer_dim)
        self.fc2 = nn.Linear(hidden_layer_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Agent:
    def __init__(self, state_dim, action_dim, hidden_layer_dim, gamma=0.9, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995, lr=0.001, maxlen=100000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.lr = lr
        self.memory = deque(maxlen=maxlen)
        self.model = DQN(state_dim, hidden_layer_dim, action_dim).cuda()
        self.target_model = DQN(state_dim, hidden_layer_dim, action_dim).cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0).cuda()
        act_values = self.model(state)
        return torch.argmax(act_values, dim=1).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).unsqueeze(0).cuda()
            next_state = torch.FloatTensor(next_state).unsqueeze(0).cuda()
            target = self.model(state).clone()
            if done:
                target[0][action] = reward
            else:
                t = self.target_model(next_state)
                target[0][action] = reward + self.gamma * torch.max(t).item()
            output = self.model(state)
            loss = self.criterion(output, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save(self, filename="snake_dqn.pth"):
        torch.save(self.model.state_dict(), filename)
        print(f"Model saved as {filename}")

    def load(self, filename="snake_dqn.pth"):
        self.model.load_state_dict(torch.load(filename))
        self.model.eval()
        print(f"Model loaded from {filename}")
