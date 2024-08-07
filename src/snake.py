import tkinter as tk
import random
import numpy as np

class SnakeGame:
    def __init__(self, width=400, height=400, cell_size=20, gui=True):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.gui = gui
        
        if self.gui:
            self.root = tk.Tk()
            self.root.title("Snake Game")
            self.root.resizable(False, False)
            
            self.canvas = tk.Canvas(self.root, width=self.width, height=self.height, bg="white")
            self.canvas.pack()
            
            self.replay_button = tk.Button(self.root, text="Replay", command=self.start_game)
            self.replay_button.pack()
        
        self.start_game()
        
        if self.gui:
            self.root.bind_all("<KeyPress>", self.on_key_press)
            self.root.mainloop()

    def start_game(self):
        self.snake = [(self.cell_size, self.cell_size), 
                      (self.cell_size, 2*self.cell_size), 
                      (self.cell_size, 3*self.cell_size)]
        self.snake_direction = "Down"
        
        self.apple_position = self.set_new_apple_position()
        
        if self.gui:
            self.canvas.delete("all")
            self.apple = self.canvas.create_rectangle(*self.apple_position, 
                                                      self.apple_position[0]+self.cell_size, 
                                                      self.apple_position[1]+self.cell_size, 
                                                      fill="red")
        self.is_game_over = False
        self.score = 0
        
        if self.gui:
            self.update_snake()
            self.root.after(100, self.move_snake)
    
    def set_new_apple_position(self):
        while True:
            x = random.randint(0, (self.width // self.cell_size) - 1) * self.cell_size
            y = random.randint(0, (self.height // self.cell_size) - 1) * self.cell_size
            if (x, y) not in self.snake:
                return x, y

    def on_key_press(self, e):
        new_direction = e.keysym
        all_directions = {"Up", "Down", "Left", "Right"}
        opposites = [{"Up", "Down"}, {"Left", "Right"}]

        if (new_direction in all_directions and
                {new_direction, self.snake_direction} not in opposites):
            self.snake_direction = new_direction

    def move_snake(self):
        if self.is_game_over:
            return
        
        head_x, head_y = self.snake[-1]
        
        if self.snake_direction == "Up":
            new_head = (head_x, head_y - self.cell_size)
        elif self.snake_direction == "Down":
            new_head = (head_x, head_y + self.cell_size)
        elif self.snake_direction == "Left":
            new_head = (head_x - self.cell_size, head_y)
        elif self.snake_direction == "Right":
            new_head = (head_x + self.cell_size, head_y)
        
        self.snake.append(new_head)
        
        if new_head == self.apple_position:
            self.apple_position = self.set_new_apple_position()
            if self.gui:
                self.canvas.coords(self.apple, *self.apple_position, 
                                   self.apple_position[0]+self.cell_size, 
                                   self.apple_position[1]+self.cell_size)
            self.score += 1
        else:
            self.snake.pop(0)
        
        if (new_head[0] < 0 or new_head[0] >= self.width or
                new_head[1] < 0 or new_head[1] >= self.height or
                len(self.snake) != len(set(self.snake))):
            self.is_game_over = True
            if self.gui:
                self.canvas.create_text(self.width // 2, self.height // 2, text="GAME OVER", fill="red", font=("Arial", 24))
        else:
            if self.gui:
                self.update_snake()
                self.root.after(100, self.move_snake)
        
    def update_snake(self):
        if self.gui:
            self.canvas.delete("snake")
            for x, y in self.snake:
                self.canvas.create_rectangle(x, y, x+self.cell_size, y+self.cell_size, fill="green", tag="snake")

    def get_state(self):
        state = np.zeros((self.width // self.cell_size, self.height // self.cell_size))
        for (x, y) in self.snake:
            state[x // self.cell_size, y // self.cell_size] = 1
        ax, ay = self.apple_position
        state[ax // self.cell_size, ay // self.cell_size] = -1
        return state.flatten()
    
    def step(self, action):
        directions = ["Up", "Down", "Left", "Right"]
        self.snake_direction = directions[action]
        self.move_snake()
        next_state = self.get_state()
        reward = 1 if self.snake[-1] == self.apple_position else -0.1
        done = self.is_game_over
        return next_state, reward, done, self.score

if __name__ == "__main__":
    game = SnakeGame(gui=True)
