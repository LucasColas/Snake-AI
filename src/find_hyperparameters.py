import optuna
from src.snake import SnakeGame
from src.DQN import Agent

def train_agent(agent, episodes=10000, batch_size=64):
    total_reward = 0
    print("...training agent...")
    for e in range(episodes):
        #print("episode : ", e)
        episode_reward = 0
        game = SnakeGame(gui=False)
        state = game.reset()
        for time in range(5000):
            action = agent.act(state)
            #print(" action : ", action, " state : ", state, " time : ", time, " episode : ", e)
            next_state, reward, done = game.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            episode_reward += reward
            if done:
                agent.update_target_model()
                break
            agent.replay(batch_size)
        if e % 100 == 0:
            print("episode : ", e, " episode_reward : ", episode_reward)
    return total_reward / episodes

def objective(trial):
    gamma = trial.suggest_float("gamma", 0.95, 0.999)
    epsilon = trial.suggest_float("epsilon", 0.1, 1.0)
    epsilon_decay = trial.suggest_float("epsilon_decay", 0.99, 0.9999)
    lr = trial.suggest_float("lr", 0.0001, 0.01, log=True)
    hidden_layer_dim = trial.suggest_int("hidden_layer_dim", 32, 512)

    agent = Agent(state_dim=8, action_dim=4, hidden_layer_dim=hidden_layer_dim, gamma=gamma, epsilon=epsilon, epsilon_decay=epsilon_decay, lr=lr)
    reward = train_agent(agent)
    return reward


