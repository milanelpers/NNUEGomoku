import numpy as np
import random
from agent import Agent


class AgentRandom(Agent):
    def place_stone(self, board):
        '''
        Plays (valid) moves at random

        Args:
            board (ndarray): The board where the stone will be placed.
            color (int): The color the agent is playing as
        
        Returns:
            pos (tuple): Position the agent chose to play
        '''
        valid_moves = np.argwhere(board == 0)
        if valid_moves.size == 0:
            raise ValueError('No valid moves left!')
        pos = tuple(random.choice(valid_moves))
        return pos