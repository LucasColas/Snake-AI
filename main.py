import pygame
import random
import sys
import numpy as np
import matplotlib.pyplot as plt
from src.agent import Agent
# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 750
SCREEN_HEIGHT = 750
CELL_SIZE = 25
GRID_SIZE = SCREEN_WIDTH // CELL_SIZE
SPEED = 60

# clock
clock = pygame.time.Clock()

# Colors
SNAKE_COLOR = (248, 168, 0)
BACKGROUND_COLOR = (104, 56, 0)
APPLE_COLOR = (1, 252, 128)
BORDER_COLOR = (232, 168, 0)
SCORE_COLOR = (248, 252, 248)

# Create the screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Snake")

font = pygame.font.Font(None, 36)
def plot(scores, mean_scores):
    
    plt.plot(scores, label='Score')
    plt.plot(mean_scores, label='Mean Score')
    plt.legend()
    plt.show()

class SnakeGameAI:
    def __init__(self):
        self.snake = [(5, 5), (4, 5), (3, 5)]
        self.direction = (1, 0)
        self.apple = self._place_apple()
        self.score = 0
        self.grow = False
        self.frame_iteration = 0

    def _place_apple(self):
        while True:
            position = (random.randint(1, GRID_SIZE - 2), random.randint(1, GRID_SIZE - 2))
            if position not in self.snake:
                return position

    def reset(self):
        self.snake = [(5, 5), (4, 5), (3, 5)]
        self.direction = (1, 0)
        self.apple = self._place_apple()
        self.score = 0
        self.grow = False
        self.frame_iteration = 0

    def get_state(self):
        head_x, head_y = self.snake[0]
        dir_l = self.direction == (-1, 0)
        dir_r = self.direction == (1, 0)
        dir_u = self.direction == (0, -1)
        dir_d = self.direction == (0, 1)

        state = [
            # Danger straight
            (dir_r and self._is_collision((head_x + 1, head_y))) or
            (dir_l and self._is_collision((head_x - 1, head_y))) or
            (dir_u and self._is_collision((head_x, head_y - 1))) or
            (dir_d and self._is_collision((head_x, head_y + 1))),

            # Danger right
            (dir_u and self._is_collision((head_x + 1, head_y))) or
            (dir_d and self._is_collision((head_x - 1, head_y))) or
            (dir_l and self._is_collision((head_x, head_y - 1))) or
            (dir_r and self._is_collision((head_x, head_y + 1))),

            # Danger left
            (dir_d and self._is_collision((head_x + 1, head_y))) or
            (dir_u and self._is_collision((head_x - 1, head_y))) or
            (dir_r and self._is_collision((head_x, head_y - 1))) or
            (dir_l and self._is_collision((head_x, head_y + 1))),

            # Move direction
            dir_l, dir_r, dir_u, dir_d,

            # Apple location
            self.apple[0] < head_x,  # apple left
            self.apple[0] > head_x,  # apple right
            self.apple[1] < head_y,  # apple up
            self.apple[1] > head_y   # apple down
        ]

        return np.array(state, dtype=int)

    def _is_collision(self, position=None):
        if position is None:
            position = self.snake[0]

        # Check if snake collides with borders
        if not (1 <= position[0] < GRID_SIZE - 1 and 1 <= position[1] < GRID_SIZE - 1):
            return True

        # Check if snake collides with itself
        if position in self.snake[1:]:
            return True

        return False

    def play_step(self, action, plot_scores, plot_mean_scores):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                plot(plot_scores, plot_mean_scores)
                sys.exit()
        self.frame_iteration += 1
        # Perform the action
        clock_wise = [(-1, 0), (0, -1), (1, 0), (0, 1)]  # left, up, right, down
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # No change
        elif np.array_equal(action, [0, 1, 0]):
            new_idx = (idx + 1) % 4  # Right turn r -> d -> l -> u
            new_dir = clock_wise[new_idx]
        else:  # [0, 0, 1]
            new_idx = (idx - 1) % 4  # Left turn l -> u -> r -> d
            new_dir = clock_wise[new_idx]

        self.direction = new_dir

        head_x, head_y = self.snake[0]
        delta_x, delta_y = self.direction
        new_head = (head_x + delta_x, head_y + delta_y)

        # Move the snake
        self.snake.insert(0, new_head)

        reward = 0
        done = False

        # Check if the snake hits the wall or itself
        if self._is_collision(new_head) or self.frame_iteration > 100*len(self.snake):
            reward = -10
            done = True
            return reward, done, self.score

        # Check if the snake eats the apple
        if new_head == self.apple:
            self.score += 1
            reward = 10
            self.apple = self._place_apple()
        else:
            self.snake.pop()

        return reward, done, self.score

    def draw(self):
        screen.fill(BACKGROUND_COLOR)
        pygame.draw.rect(screen, BORDER_COLOR, pygame.Rect(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT), CELL_SIZE)

        # Draw the snake
        for x, y in self.snake:
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, SNAKE_COLOR, rect)

        # Draw the apple
        rect = pygame.Rect(self.apple[0] * CELL_SIZE, self.apple[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, APPLE_COLOR, rect)

        # Display the score
        text = font.render(f"Score: {self.score}", True, SCORE_COLOR)
        screen.blit(text, (10, 10))

        pygame.display.flip()


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    # load model
    #agent.model.load()
    game = SnakeGameAI()
    run = True
    while run:

        

        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move, plot_scores, plot_mean_scores)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)
        game.draw()
        clock.tick(SPEED)
        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
    #plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()
