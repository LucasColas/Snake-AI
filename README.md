# Snake-AI

Deep Q learning that learns to play Snake !

<img src="https://github.com/LucasColas/Snake-AI/blob/main/img/sneak%20peak.jpg" width=30% height=30%>

Results of the score of Deep Q Learning through episodes
<img src="https://github.com/LucasColas/Snake-AI/blob/main/img/Premier%20res.png" width=30% height=30%>
x-axis : episodes
y-axis : score 

An episode is ended when the snake collides with itself or a wall.

## Run the code - Prerequisites 
* [Python >= 3.10](https://www.python.org/)

Using a virtual environment is recommended. 

## How to play

* git clone this repository
  ```bash
  git clone git@github.com:LucasColas/Maze-Solver.git
  ```
* go to the directory
  ```bash
  cd Maze-Solver
  ```

Install the dependencies

```
bash
pip install requirements.txt
```

* run `game_AI.py`. The game with the AI learning snake will be displayed.. The best model will be saved.
* Enjoy !

## More explanation

game_AI.py is the entry point. The folder called src contains the code of the project. `agent.py` contains the code related to the deep q learning algorithm. You can tweak the hyperparameters in this file.
If you run `game_AI.py` it will train the AI. It launches a pygame window with the AI playing the snake. in `game_AI.py` there's function called train. You can change (uncomment) the code to load an existing model if you want. If you have no model, a model will be saved by the code. Every time the AI beats the record the model is saved.
I used Deep Q Learning. There's not a predined number of episodes. You close the game whenever you want.

The neural network is a multi layer perceptron with one hidden layer. 

You can also press "m" if you want to increase the speed of the rendering of the game (hence you increase the speed of training). Press "l" if you want to reduce the speed (it's good to see your AI playing snake).
