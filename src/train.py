import optuna
import matplotlib.pyplot as plt
from src.snake import SnakeGame
from src.DQN import Agent

def train_agent(agent, episodes=100, n_steps=2000, batch_size=1000):
    total_reward = 0
    best_score = 0
    plot_scores = []
    plot_rewards = []
    print("...training agent...")
    for e in range(episodes):
        #print("episode : ", e)
        episode_reward = 0
        game = SnakeGame(gui=False)
        state = game.reset()
        while True:
            action = agent.act(state)
            #print(" action : ", action, " state : ", state, " time : ", time, " episode : ", e)
            next_state, reward, done, score = game.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            episode_reward += reward
            if score > best_score:
                best_score = score
                print("best score : ", best_score)
                agent.save(f'snake_dqn_{best_score}.pth')
            if done:
                agent.update_target_model()
                plot_scores.append(score)
                plot_rewards.append(episode_reward)
                break
            agent.replay(batch_size)
        if e and e % 100 == 0:
            print("episode : ", e, " mean episode_reward : ", total_reward / e)
            print("best score : ", best_score)
            agent.save(f'snake_dqn_{best_score}_episode_{e}.pth')

    print("mean episode_reward : ", total_reward / episodes)
    plot_scores(plot_scores, plot_rewards)
    print("...training complete...")

    return total_reward / episodes

def objective(trial):
    print("...finding best hyperparameters...")
    gamma = trial.suggest_float("gamma", 0.95, 0.999)
    epsilon = trial.suggest_float("epsilon", 0.1, 1.0)
    epsilon_decay = trial.suggest_float("epsilon_decay", 0.99, 0.9999)
    lr = trial.suggest_float("lr", 0.0001, 0.01, log=True)
    hidden_layer_dim = trial.suggest_int("hidden_layer_dim", 32, 512)

    agent = Agent(state_dim=8, action_dim=4, hidden_layer_dim=hidden_layer_dim, gamma=gamma, epsilon=epsilon, epsilon_decay=epsilon_decay, lr=lr)
    reward = train_agent(agent)
    return reward

def plot_scores(plot_scores, plot_rewards):
    
    plt.plot(plot_scores)
    plt.plot(plot_rewards)
    plt.title("Snake Game")
    plt.xlabel("Episodes")
    plt.ylabel("Values")
    plt.legend(["Scores", "Rewards"])
    plt.show()
