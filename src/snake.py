import tkinter as tk
import random

class SnakeGame(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("Snake Game")
        self.resizable(False, False)
        
        self.canvas = tk.Canvas(self, width=400, height=400, bg="white")
        self.canvas.pack()
        
        self.snake = [(20, 20), (20, 30), (20, 40)]
        self.snake_direction = "Down"
        
        self.apple_position = self.set_new_apple_position()
        self.apple = self.canvas.create_rectangle(*self.apple_position, self.apple_position[0]+10, self.apple_position[1]+10, fill="red")
        
        self.bind_all("<KeyPress>", self.on_key_press)
        self.update_snake()
        
        self.after(100, self.move_snake)
        
    def set_new_apple_position(self):
        while True:
            x = random.randint(0, 39) * 10
            y = random.randint(0, 39) * 10
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
        head_x, head_y = self.snake[-1]
        
        if self.snake_direction == "Up":
            new_head = (head_x, head_y - 10)
        elif self.snake_direction == "Down":
            new_head = (head_x, head_y + 10)
        elif self.snake_direction == "Left":
            new_head = (head_x - 10, head_y)
        elif self.snake_direction == "Right":
            new_head = (head_x + 10, head_y)
        
        self.snake.append(new_head)
        
        if new_head == self.apple_position:
            self.apple_position = self.set_new_apple_position()
            self.canvas.coords(self.apple, *self.apple_position, self.apple_position[0]+10, self.apple_position[1]+10)
        else:
            self.snake.pop(0)
        
        if (new_head[0] < 0 or new_head[0] >= 400 or
                new_head[1] < 0 or new_head[1] >= 400 or
                len(self.snake) != len(set(self.snake))):
            self.game_over()
        else:
            self.update_snake()
            self.after(100, self.move_snake)
        
    def update_snake(self):
        self.canvas.delete("snake")
        for x, y in self.snake:
            self.canvas.create_rectangle(x, y, x+10, y+10, fill="green", tag="snake")
            
    def game_over(self):
        self.canvas.create_text(200, 200, text="GAME OVER", fill="red", font=("Arial", 24))

