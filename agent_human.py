import numpy as np
from agent import Agent


class AgentHuman(Agent):
    def place_stone(self, board, color):
        '''
        Asks the user to input a move

        Args:
            board (ndarray): The board object where the stone will be placed.
            color (int): The color the agent is playing as

        Returns:
            pos (tuple): Position the agent chose to play
        '''
        while True:
            pos = input('Enter the row and column number (0-indexed, separated by a space) of where you want your stone to be placed: ')
            pos_split = pos.split(' ')
            if pos_split[0] == pos:
                print(f'{pos} had no spaces to separate row and col')
                continue
            try:
                pos_int = (int(pos_split[0]), int(pos_split[1]))
            except:
                print(f'{pos} is not a valid position')
                continue
            if pos_int[0] < 0 or pos_int[0] > 14 or pos_int[1] < 0 or pos_int[1] > 14:
                print(f'{pos} is outside the board')
                continue
            if board[pos_int[0], pos_int[1]] != 0:
                print(f'{pos} is not a valid move as there is already a stone there')
                continue
            else:
                pos = (pos_int[0], pos_int[1])
                break
        return pos