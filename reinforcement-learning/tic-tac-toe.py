import numpy as np
import random
from tqdm import tqdm

class TTTAgent:
    def __init__(self, lr=.7, gamma=.8, epsilon=.3, episodes=15000):
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.episodes = episodes

    def _create_board(self):
        self.is_over = False
        self.board = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11])
        self._check_or_add_state()

    def _check_or_add_state(self):
        self.state = str(list(self.board))
        if self.state_dict.get(self.state) is None:
            # adding to state dictionary
            self.state_dict[self.state] = np.zeros(len(self.board[self.board > 2]))
        self.action_space = self.state_dict[self.state] #lis of available actions
        if len(self.action_space) == 0:
            self.action_space = np.array([0])

    def _play_random(self, player):
        #Exploration in q-learning
        random_ind = random.choice(np.where(self.board > 2)[0])
        self.board[random_ind] = player
        self._check_or_add_state()

    def _act(self, possible_random=True):
        #agent acting

        if possible_random and (np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) <= (self.epsilon * 10)):
            self.action = np.argmax(np.random.rand(1, len(self.action_space)))
        else:
            if possible_random:
                self.action = np.argmax(
                    self.action_space + np.random.randn(1, len(self.action_space)) * (1. / (self.i + 1)))
            else:
                self.action = np.argmax(self.action_space)
        self.board[self.board[self.board > 2][self.action] - 3] = 1
        self._check_or_add_state()

    def _step(self):
        #next step in the game

        self._is_finished()
        if self.winner == 0:
            self._play_random(2)
            self._is_finished()

    def _update_state_dict(self, current_state):
        #bellman equation
        self.state_dict[current_state][self.action] = (1 - self.lr) * self.state_dict[current_state][self.action] \
                                                      + self.lr * (self.reward \
                                                                   + self.gamma * np.max(self.action_space))

    def _is_finished(self):
        #Game over?

        self.reward = 0
        self.winner = 0

        if len(self.board[self.board > 2]) == 0:
            # Then it's a draw
            self.winner = 3
            self.is_over = True

        shaped_board = self.board.reshape(3, 3) #Turning it into a matrix
        for i in range(3):
            set_row = set(shaped_board[i])
            if len(set_row) == 1:
                self.winner = list(set_row)[0]
            set_col = set(shaped_board[:, i])
            if len(set_col) == 1:
                self.winner = list(set_col)[0]

        set_left_diag = set(shaped_board.reshape(3, 3).diagonal(0))
        if len(set_left_diag) == 1:
            self.winner = list(set_left_diag)[0]

        set_right_diag = set(np.flipud(shaped_board.reshape(3, 3)).diagonal(0))
        if len(set_right_diag) == 1:
            self.winner = list(set_right_diag)[0]

        if self.winner == 2:
            self.reward = -1 #A lost game
            self.is_over = True
        elif self.winner == 1:
            self.reward = 1 #A won game
            self.is_over = True

    def learn(self, ensemble=False):
        #learning for one agent

        self.state_dict = {}
        self.rewards_lists = [np.zeros(self.episodes), np.zeros(self.episodes)]
        for i in range(self.episodes):
            self.i = i
            for j in range(2):
                self._create_board()
                if j == 1:
                    # it's the user's turn

                    self._step()
                for _ in range(9):
                    current_state = self.state
                    self._act()
                    self._step()
                    self._update_state_dict(current_state)
                    if self.is_over == True:
                        break
                self.rewards_lists[j][i] += self.reward
        if ensemble:
            return self.state_dict

    def learn_multi(self, iterations):
        #various agents learning

        self.state_dict_ensemble = {}
        for i in tqdm(range(iterations)):
            state_dict = self.learn(ensemble=True)
            for state, actions in state_dict.items():
                if self.state_dict_ensemble.get(state) is not None:
                    self.state_dict_ensemble[state] += actions
                else:
                    self.state_dict_ensemble[state] = actions
        self.state_dict = self.state_dict_ensemble

    def _display_board(self):
        """Display current board.
        """
        new_board = []
        for b in self.board:
            if b == 1:
                new_board.append('X')
            elif b == 2:
                new_board.append('O')
            else:
                new_board.append(' ')
        print("""
     {} | {} | {} 
    ---|---|---
     {} | {} | {} 
    ---|---|---
     {} | {} | {} 
    """.format(new_board[0], new_board[1], new_board[2],
               new_board[3], new_board[4], new_board[5],
               new_board[6], new_board[7], new_board[8]))

    def _play_computer(self):
        #agent playing

        print('\nComputer made a move:')
        self._act(possible_random=False)

    def _play_human(self):
        #user is playing

        print('\nYou made a move:')
        ind = int(input('Select the index of your move: '))
        self.board[ind] = 2
        self._check_or_add_state()

    def play(self):
        """Play a game against the agent.
        """
        self._create_board()
        self._display_board()
        whos_first = input('Do you want to go first? (y/n): ')
        for i in range(9):
            if i % 2 == 0:
                if whos_first == 'y':
                    self._play_human()
                else:
                    self._play_computer()
            else:
                if whos_first == 'y':
                    self._play_computer()
                else:
                    self._play_human()
            self._display_board()
            self._is_finished()
            if self.is_over == True:
                print("Game Over!")
                break


ttt = TTTAgent()
ttt.learn_multi(50)

ttt.play()
