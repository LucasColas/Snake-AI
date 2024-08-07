import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random 
from src.snake import SnakeGame

class DQN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class DeepQNetwork:
    def __init__(self, state_dim, action_dim, hidden_dim=512, gamma=0.99, lr=0.001, batch_size=64, replay_memory_size=100000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        
        self.replay_memory = deque(maxlen=replay_memory_size)
        self.model = DQN(state_dim, hidden_dim, action_dim)
        self.target_model = DQN(state_dim, hidden_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def choose_action(self, state, epsilon):
        if np.random.rand() <= epsilon:
            return np.random.randint(self.action_dim)
        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.model(state)
            return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_memory.append((state, action, reward, next_state, done))

    def train_step(self):
        if len(self.replay_memory) < self.batch_size:
            return

        batch = random.sample(self.replay_memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_model(next_states).max(1)[0]
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = self.loss_fn(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def train_snake(num_episodes=1000, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, target_update_freq=10):
    game = SnakeGame(gui=False)
    state_dim = (game.width // game.cell_size) * (game.height // game.cell_size)
    action_dim = 4  # Up, Down, Left, Right
    dqn = DeepQNetwork(state_dim, action_dim)

    epsilon = epsilon_start
    for episode in range(num_episodes):
        state = game.get_state()
        total_reward = 0
        done = False

        while not done:
            action = dqn.choose_action(state, epsilon)
            next_state, reward, done, score = game.step(action)
            dqn.store_transition(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            dqn.train_step()

        if epsilon > epsilon_end:
            epsilon *= epsilon_decay

        if episode % target_update_freq == 0:
            dqn.update_target_model()

        if episode % 1000 == 0:
            print(f"Episode {episode}, Score: {score}, Total Reward: {total_reward}, Epsilon: {epsilon}")

    return dqn

def test_snake(dqn):
    game = SnakeGame(gui=True)
    state = game.get_state()
    done = False

    while not done:
        action = dqn.choose_action(state, epsilon=0)  # No exploration during testing
        next_state, _, done, _ = game.step(action)
        state = next_state
        game.root.update_idletasks()
        game.root.update()
        game.root.after(100)