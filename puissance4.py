import py5
import random
import neat
import math
import os
import pickle
import numpy as np

h, v = 7, 6
size = 60
turn = 0
win = -1
singleplayer = False

local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, "neat_config.txt")

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)

with open(os.path.join(local_dir, "best.pickle"), "rb") as f:
    winner = pickle.load(f)

winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

#0 = pas de jeton ; 1=jeton du joueur 1 ; 2=jeton du joueur 2
grid = np.array([[0 for i in range(h)] for j in range(v)])

class Button:
    def __init__(self, x, y, width, height, label):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.label = label

    def display(self):
        py5.stroke(0)
        py5.stroke_weight(2)
        py5.fill(200)
        py5.rect(self.x, self.y, self.width, self.height)
        py5.fill(0)
        py5.text_align(py5.CENTER, py5.CENTER)
        py5.text(self.label, self.x + self.width / 2, self.y + self.height / 2)

def draw_grid():
    global h, v, size
    py5.fill(0)
    py5.stroke_weight(1)
    for i in range(1,h):
        py5.line(i*size, 0, i*size, h*size)
    for i in range(1,v):
        py5.line(0, i*size, (v+1)*size, i*size)

def update_grid(grid):
    py5.fill(0)
    py5.stroke_weight(1)
    for i in range(v):
        for j in range(h):
            if grid[i][j] == 1:
                py5.fill(255, 0, 0)
                py5.circle(j*size + size//2, i*size + size//2, size-5)
            elif grid[i][j] == 2:
                py5.fill(0, 0, 255)
                py5.circle(j*size + size//2, i*size + size//2, size-5)

def check_win(grid):
    # Check horizontal
    py5.stroke_weight(6)
    for row in range(v):
        for col in range(h - 3):
            if grid[row][col] != 0 and grid[row][col] == grid[row][col + 1] == grid[row][col + 2] == grid[row][col + 3]:
                # Draw winning line horizontally
                py5.stroke(255, 255, 0)  # Yellow color
                py5.line(col * size + size / 2, row * size + size / 2, (col + 3) * size + size / 2, row * size + size / 2)
                if grid[row][col] == 1:
                    return 1
                elif grid[row][col] == 2:
                    return 2

    # Check vertical
    for row in range(v - 3):
        for col in range(h):
            if grid[row][col] != 0 and grid[row][col] == grid[row + 1][col] == grid[row + 2][col] == grid[row + 3][col]:
                # Draw winning line vertically
                py5.stroke(255, 255, 0)  # Yellow color
                py5.line(col * size + size / 2, row * size + size / 2, col * size + size / 2, (row + 3) * size + size / 2)
                if grid[row][col] == 1:
                    return 1
                elif grid[row][col] == 2:
                    return 2

    # Check diagonal (positive slope)
    for row in range(v - 3):
        for col in range(h - 3):
            if grid[row][col] != 0 and grid[row][col] == grid[row + 1][col + 1] == grid[row + 2][col + 2] == grid[row + 3][col + 3]:
                 # Draw winning line diagonally (positive slope)
                py5.stroke(255, 255, 0)  # Yellow color
                py5.line(col * size + size / 2, row * size + size / 2, (col + 3) * size + size / 2, (row + 3) * size + size / 2)
                if grid[row][col] == 1:
                    return 1
                elif grid[row][col] == 2:
                    return 2

    # Check diagonal (negative slope)
    for row in range(3, v):
        for col in range(h - 3):
            if grid[row][col] != 0 and grid[row][col] == grid[row - 1][col + 1] == grid[row - 2][col + 2] == grid[row - 3][col + 3]:
                # Draw winning line diagonally (negative slope)
                py5.stroke(255, 255, 0)  # Yellow color
                py5.line(col * size + size / 2, row * size + size / 2, (col + 3) * size + size / 2, (row - 3) * size + size / 2)
                if grid[row][col] == 1:
                    return 1
                elif grid[row][col] == 2:
                    return 2

    return 0  # If no winner is found

def is_column_full(grid, column):
    for row in range(len(grid)):
        if grid[row][column] == 0:
            return False  # Column is not full
    return True  # Column is full


def play(grid, column, player):
    if player == 1:
        for i in range(v):
            if grid[v-i-1][column] == 0:
                grid[v-i-1][column] = 1
                break
    elif player == 2:
        for i in range(v):
            if grid[v-i-1][column] == 0:
                grid[v-i-1][column] = 2
                break

def draw_centered_text(text, font_size=32):
    py5.text_align(py5.CENTER, py5.CENTER)
    py5.text_size(font_size)
    py5.fill(0)
    py5.text(text, py5.width / 2, py5.height / 2)

def reset_game():
    global grid, win, h, v
    grid = [[0 for i in range(h)] for j in range(v)]
    win = 0
    py5.background(200)
    draw_grid()

def key_pressed():
    global win
    if py5.key == 'r' or py5.key == 'R':
        win = -1

def setup():
    py5.size(size*h, size*v)
    py5.frame_rate(15)
    py5.background(200)
    draw_grid()

def mouse_pressed():
    global turn, win, singleplayer
    #Multiplayer
    if win == 0 and singleplayer == False:
        mouse_x = py5.mouse_x//size
        mouse_y = py5.mouse_y//size
        if turn%2 == 0:
            if not is_column_full(grid, mouse_x):
                play(grid, mouse_x, 1)
                turn += 1
        else:
            if not is_column_full(grid, mouse_x):
                play(grid, mouse_x, 2)
                turn += 1

    #Singleplayer
    elif win == 0 and singleplayer == True:
        mouse_x = py5.mouse_x//size
        mouse_y = py5.mouse_y//size
        
        if is_column_full(grid, mouse_x):
            return
        play(grid, mouse_x, 1)
        win = check_win(grid)

        if win == 0:
            output2 = winner_net.activate(np.array(grid).flatten())
            decision2 = output2.index(max(output2))

            while is_column_full(grid, decision2) == True:
                decision2 = IA_puissance4.compute_best_move(grid)
            play(grid, decision2, 2)

    
    #Check for button click
    elif win == -1:
        if singleplayer_button.x <= py5.mouse_x <= singleplayer_button.x + singleplayer_button.width and \
            singleplayer_button.y <= py5.mouse_y <= singleplayer_button.y + singleplayer_button.height:

            reset_game()
            singleplayer = True

        elif multiplayer_button.x <= py5.mouse_x <= multiplayer_button.x + multiplayer_button.width and \
                multiplayer_button.y <= py5.mouse_y <= multiplayer_button.y + multiplayer_button.height:
            reset_game()
            singleplayer = False


def draw():
    global win
    if win == -1:
        py5.background(255)
        py5.fill(0)
        py5.text_size(32)
        py5.text("Connect 4", py5.width / 2, 50)

        singleplayer_button.display()
        multiplayer_button.display()
    elif win == 0:
        update_grid(grid)
        win = check_win(grid)
    elif win == 1:
        py5.background(200)
        draw_centered_text("Player 1 won")
        py5.text_size(16)
        py5.text("Press R to restart", py5.width/2, size*v/2 + 30)
        singleplayer = False
        pass
    elif win ==2:
        py5.background(200)
        draw_centered_text("Player 2 won")
        py5.text_size(16)
        py5.text("Press R to restart", py5.width/2, size*v/2 + 30)
        singleplayer = False
        pass


singleplayer_button = Button(100, 120, 200, 50, "Singleplayer")
multiplayer_button = Button(100, 190, 200, 50, "Multiplayer")

if __name__ =='__main__':
    py5.run_sketch()