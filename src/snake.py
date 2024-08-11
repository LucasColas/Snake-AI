import random
import numpy as np
import tkinter as tk

class SnakeGame:
    def __init__(self, grid_size=20, cell_size=20, gui=False):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.gui = gui

        if self.gui:
            self.window = tk.Tk()
            self.window.title("Snake Game")
            self.window.resizable(False, False)
            self.canvas = tk.Canvas(self.window, bg="white", height=grid_size*cell_size, width=grid_size*cell_size)
            self.canvas.pack()
            self.window.bind("<KeyPress>", self.on_key_press)
        
        self.reset()

    def reset(self):
        self.snake = [(5, 5), (5, 4), (5, 3)]
        self.snake_direction = (0, 1)
        self.food = self.place_food()
        self.done = False
        self.score = 0
        return self.get_state()

    def place_food(self):
        while True:
            food = (random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1))
            if food not in self.snake:
                return food

    def step(self, action):
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Up, Down, Left, Right
        self.snake_direction = directions[action]
        head_x, head_y = self.snake[0]
        move_x, move_y = self.snake_direction
        new_head = (head_x + move_x, head_y + move_y)
        self.snake = [new_head] + self.snake[:-1]

        if new_head == self.food:
            self.snake.append(self.snake[-1])
            self.food = self.place_food()
            self.score += 1
            reward = 10
        else:
            reward = 0
        
        if self.is_collision(new_head):
            self.done = True
            reward = -10

        return self.get_state(), reward, self.done, self.score

    def is_collision(self, position):
        x, y = position
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            #print("collision with wall")
            return True
        if position in self.snake[1:]:
            #print("collision with itself")
            return True
        return False

    def get_state(self):
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        return np.array([
            head_x, head_y,
            food_x, food_y,
            self.snake_direction[0], self.snake_direction[1],
            (head_x - food_x) / self.grid_size,
            (head_y - food_y) / self.grid_size
        ])

    def on_key_press(self, e):
        directions = {"Up": 0, "Down": 1, "Left": 2, "Right": 3}
        if e.keysym in directions:
            self.snake_direction = directions[e.keysym]

    def update_canvas(self):
        self.canvas.delete(tk.ALL)
        for x, y in self.snake:
            self.canvas.create_rectangle(x*self.cell_size, y*self.cell_size,
                                         (x+1)*self.cell_size, (y+1)*self.cell_size, fill="green")
        food_x, food_y = self.food
        self.canvas.create_rectangle(food_x*self.cell_size, food_y*self.cell_size,
                                     (food_x+1)*self.cell_size, (food_y+1)*self.cell_size, fill="red")

    def game_loop(self, agent):
        if not self.done:
            state = self.get_state()
            action = agent.act(state)
            
            next_state, reward, done, _ = self.step(action)
            agent.remember(state, action, reward, next_state, done)
            self.update_canvas()
            if done:
                self.reset()
            self.window.after(100, self.game_loop, agent)
    
    def run_with_gui(self, agent):
        self.update_canvas()
        self.window.after(100, self.game_loop, agent)
        self.window.mainloop()


if __name__ == "__main__":
    game = SnakeGame(gui=True)
    game.run_with_gui(None)