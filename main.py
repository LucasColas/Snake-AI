from src.snake import SnakeGame
from src.DQN import train_snake, test_snake

def main():
    trained_dqn = train_snake(num_episodes=1000000)
    test_snake(trained_dqn)

if __name__ == "__main__":
    main()