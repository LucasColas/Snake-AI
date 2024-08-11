import optuna
from src.train import objective, train_agent
from src.DQN import Agent
from src.snake import SnakeGame

def main():
    """study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    print("Best hyperparameters: ", study.best_params)"""
    agent = Agent(state_dim=8, action_dim=4, hidden_layer_dim=256)
    train_agent(agent, episodes=1000)
    #agent.save()

    agent.load("snake_dqn.pth")
    game = SnakeGame(gui=True)
    game.run_with_gui(agent)

if __name__ == "__main__":
    main()
