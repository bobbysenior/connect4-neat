import neat
import numpy as np
import neat
import os
import pickle

WIDTH = 7
HEIGHT = 6
EMPTY = 0
PLAYER1 = 1
PLAYER2 = 2

local_dir = os.path.dirname(__file__)

# Return 69 mean a column is full

class Connect4:
    def __init__(self):
        self.turn = 0
        self.board = np.array([[0 for _ in range(WIDTH)] for _ in range(HEIGHT)])
        self.win = 0

    def board_to_vector(self):
            return self.board.flatten()

    def play(self, player, column):
        if self.is_column_full(self.board, column) == True:
            return 69
        if player == PLAYER1:
            for i in range(HEIGHT):
                if self.board[HEIGHT-i-1][column] == 0:
                    self.board[HEIGHT-i-1][column] = 1
                    return 200
        elif player == PLAYER2:
            for i in range(HEIGHT):
                if self.board[HEIGHT-i-1][column] == 0:
                    self.board[HEIGHT-i-1][column] = 2
                    return 200

    def is_column_full(self, board, column):
        for row in range(len(self.board)):
            if self.board[row][column] == 0:
                return False  # Column is not full
        return True  # Column is full

    def check_win(self):
        # Check horizontally
        for row in range(HEIGHT):
            for col in range(WIDTH - 3):
                if self.board[row][col] == self.board[row][col + 1] == self.board[row][col + 2] == self.board[row][col + 3] != EMPTY:
                    return self.board[row][col]

        # Check vertically
        for col in range(WIDTH):
            for row in range(HEIGHT - 3):
                if self.board[row][col] == self.board[row + 1][col] == self.board[row + 2][col] == self.board[row + 3][col] != EMPTY:
                    return self.board[row][col]

        # Check diagonally (positive slope)
        for row in range(HEIGHT - 3):
            for col in range(WIDTH - 3):
                if self.board[row][col] == self.board[row + 1][col + 1] == self.board[row + 2][col + 2] == self.board[row + 3][col + 3] != EMPTY:
                    return self.board[row][col]

        # Check diagonally (negative slope)
        for row in range(3, HEIGHT):
            for col in range(WIDTH - 3):
                if self.board[row][col] == self.board[row - 1][col + 1] == self.board[row - 2][col + 2] == self.board[row - 3][col + 3] != EMPTY:
                    return self.board[row][col]

        return 0  # No winner yet

    def is_board_full(self):
        board_full = True
        for i in range(WIDTH):
            if self.is_column_full(self.board, i) == False:
                board_full == False
        return False

    def loop(self):
        if self.turn % 2 == 0:
            p1_move = int(input("Enter p1 move : "))
            while self.play(PLAYER1, p1_move) == 69:
                print("Column is full, pick another")
                p1_move = int(input("Enter p1 move : "))
            self.turn += 1
            self.win = self.check_win()
            self.draw_board()
        elif self.turn % 2 == 1:
            p2_move = int(input("Enter p2 move : "))
            while self.play(PLAYER2, p2_move) == 69:
                print("Column is full, pick another")
                p2_move = int(input("Enter p1 move : "))
            self.turn += 1
            self.win = self.check_win()
            self.draw_board()
        
    def prevent_alignment(self, player, opponent):
        player_score = 0
        # Check if opponent is close to aligning four tokens horizontally
        for row in range(HEIGHT):
            for col in range(WIDTH - 3):
                if self.board[row][col:col + 4].tolist() == [opponent, opponent, opponent, player] or self.board[row][col:col + 4].tolist() == [player, opponent, opponent, opponent]:
                        player_score += 3
        # Check if opponent is close to aligning four tokens vertically
        for col in range(WIDTH):
            for row in range(HEIGHT - 3):
                if [self.board[row + i][col] for i in range(4)] == [opponent, opponent, opponent, player] or [self.board[row + i][col] for i in range(4)] == [player, opponent, opponent, opponent]:
                        player_score += 3
        # Check if opponent is close to aligning four tokens diagonally (positive slope)
        for row in range(HEIGHT - 3):
            for col in range(WIDTH - 3):
                if [self.board[row + i][col + i] for i in range(4)] == [opponent, opponent, opponent, player] or [self.board[row + i][col + i] for i in range(4)] == [player, opponent, opponent, opponent]:
                        player_score += 3
        # Check if opponent is close to aligning four tokens diagonally (negative slope)
        for row in range(3, HEIGHT):
            for col in range(WIDTH - 3):
                if [self.board[row - i][col + i] for i in range(4)] == [opponent, opponent, opponent, player] or [self.board[row - i][col + i] for i in range(4)] == [player, opponent, opponent, opponent]:
                        player_score += 3
        return player_score

    def draw_board(self):
        print(self.board)

    def run(self):
        while self.win == 0 and self.is_board_full() == False:
            self.loop()
        print(f"Player {self.win} won !")

    def train_ai(self, genome1, genome2, config):
        net1 = neat.nn.FeedForwardNetwork.create(genome1, config)
        net2 = neat.nn.FeedForwardNetwork.create(genome2, config)
        column_full_counter = 0
        
        while self.win == 0 and self.is_board_full() == False:
            if column_full_counter > 0:
                break

            if self.turn % 2 == 0:
                output1 = net1.activate(self.board_to_vector())
                decision1 = output1.index(max(output1))
                while self.play(PLAYER1, decision1) == 69:
                    if column_full_counter > 0:
                        break
                    #print("Column is full, pick another")
                    column_full_counter += 1
                    output1 = net1.activate(self.board_to_vector())
                    decision1 = output1.index(max(output1))
                self.turn += 1
                self.win = self.check_win()
                #genome1.fitness += self.prevent_alignment(PLAYER1, PLAYER2)
                #self.draw_board()

            elif self.turn % 2 == 1:
                output2 = net2.activate(self.board_to_vector())
                decision2 = output2.index(max(output2))
                while self.play(PLAYER2, decision2) == 69:
                    if column_full_counter > 0:
                        break
                    #print("Column is full, pick another")
                    column_full_counter += 1
                    output2 = net2.activate(self.board_to_vector())
                    decision2 = output2.index(max(output2))
                self.turn += 1
                self.win = self.check_win()
                #genome2.fitness += self.prevent_alignment(PLAYER2, PLAYER1)
                #self.draw_board()

        #Calcul des fitness
        if column_full_counter > 0:
            if self.turn % 2 == 1:
                genome1.fitness -= 150
            elif self.turn % 2 == 0:
                genome2.fitness -= 150
        
        if self.win == 1:
            genome1.fitness += 50
            genome2.fitness -= 50
        elif self.win == 2:
            genome2.fitness += 50
            genome1.fitness -= 50

    def test_ai(self, net):
        while self.win == 0 and self.is_board_full() == False:
            if self.turn % 2 == 0:
                p1_move = int(input("Enter p1 move : "))
                while self.play(PLAYER1, p1_move) == 69:
                    print("Column is full, pick another")
                    p1_move = int(input("Enter p1 move : "))
                self.turn += 1
                self.win = self.check_win()
                self.draw_board()
            elif self.turn % 2 == 1:
                output2 = net.activate(self.board_to_vector())
                decision2 = output2.index(max(output2))
                self.play(PLAYER2, decision2)
                self.turn += 1
                self.win = self.check_win()
                self.draw_board()
        print(f"Player {self.win} won !")

game = Connect4()

def eval_genomes(genomes, config):
    for i, (genome_id1, genome1) in enumerate(genomes):
        if i == len(genomes) - 1:
            break
        genome1.fitness = 0
        for genome_id2, genome2 in genomes[i+1:]:
            genome2.fitness = 0 if genome2.fitness == None else genome2.fitness
            game = Connect4()
            game.train_ai(genome1, genome2, config)

def run_neat(config):
    checkpoint_path = os.path.join(local_dir, 'checkpoints/neat-checkpoint-2303')
    p = neat.Checkpointer.restore_checkpoint(checkpoint_path)
    #p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(generation_interval=1, filename_prefix=os.path.join(local_dir, 'checkpoints/neat-checkpoint-')))

    winner = p.run(eval_genomes, 1)
    with open(os.path.join(os.path.dirname(__file__), "best.pickle"), "wb") as f:
        pickle.dump(winner, f)

def test_ai(config):
    with open(os.path.join(os.path.dirname(__file__), "best.pickle"), "rb") as f:
        winner = pickle.load(f)
    
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    game = Connect4()
    game.test_ai(winner_net)



if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "neat_config.txt")

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    run_neat(config)
    print("Entrainement terminé !")
    #test_ai(config)