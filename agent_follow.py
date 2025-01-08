import numpy as np
import random
from agent import Agent
from eval_numba import x_in_row_finder, chose_empty_from_half_open


class AgentFollowOpponent(Agent):
    def place_stone(self, board, color):
        '''
        Plays "around" the opponent, i.e. tries to block the opponent from winning by placing next to their stones.
        When it doesn't need to block, it will prioritize playing in the middle of the board, as there are potentially more free positions around it.
        Least favorite plays: corners
        Second least favorite plays: edges
        Favorite plays: middle

        Args:
            board (ndarray): The board where the stone will be placed.
            color (int): The color the agent is playing as

        Returns:
            pos (ndarray): Position the agent chose to play, shaped as (2,)
        '''
        default_value = np.array([-1, -1, -1, -1], dtype=np.int8)
        half_open_xs_black, open_xs_black, half_open_xs_white, open_xs_white = x_in_row_finder(board)
        half_open_4s_black = half_open_xs_black[2][~np.all(half_open_xs_black[2] == default_value, axis=1)]
        half_open_3s_black = half_open_xs_black[1][~np.all(half_open_xs_black[1] == default_value, axis=1)]
        half_open_2s_black = half_open_xs_black[0][~np.all(half_open_xs_black[0] == default_value, axis=1)]
        open_4s_black = open_xs_black[2][~np.all(open_xs_black[2] == default_value, axis=1)]
        open_3s_black = open_xs_black[1][~np.all(open_xs_black[1] == default_value, axis=1)]
        open_2s_black = open_xs_black[0][~np.all(open_xs_black[0] == default_value, axis=1)]
        half_open_4s_white = half_open_xs_white[2][~np.all(half_open_xs_white[2] == default_value, axis=1)]
        half_open_3s_white = half_open_xs_white[1][~np.all(half_open_xs_white[1] == default_value, axis=1)]
        half_open_2s_white = half_open_xs_white[0][~np.all(half_open_xs_white[0] == default_value, axis=1)]
        open_4s_white = open_xs_white[2][~np.all(open_xs_white[2] == default_value, axis=1)]
        open_3s_white = open_xs_white[1][~np.all(open_xs_white[1] == default_value, axis=1)]
        open_2s_white = open_xs_white[0][~np.all(open_xs_white[0] == default_value, axis=1)]
        
        for x in range(4, 1, -1):
            if color == 1:
                if x == 4:
                    if len(open_4s_white) > 0:
                        start_pos_1, end_pos_1, start_pos_2, end_pos_2 = random.choice(open_4s_white)
                        return np.array([start_pos_1, end_pos_1])
                    if len(half_open_4s_white) > 0:
                        pos = random.choice(half_open_4s_white)
                        return chose_empty_from_half_open(board, pos)
                if x == 3:
                    if len(open_3s_white) > 0:
                        pos = random.choice(open_3s_white)
                        return pos
                    if len(half_open_3s_white) > 0:
                        pos = random.choice(half_open_3s_white)
                        return chose_empty_from_half_open(board, pos)
                if x == 2:
                    if len(open_2s_white) > 0:
                        pos = random.choice(open_2s_white)
                        return pos
                    if len(half_open_2s_white) > 0:
                        pos = random.choice(half_open_2s_white)
                        return chose_empty_from_half_open(board, pos)
            else:
                if x == 4:
                    if len(open_4s_black) > 0:
                        start_pos_1, end_pos_1, start_pos_2, end_pos_2 = random.choice(open_4s_black)
                        return np.array([start_pos_1, end_pos_1])
                    if len(half_open_4s_black) > 0:
                        pos = random.choice(half_open_4s_black)
                        return chose_empty_from_half_open(board, pos)
                if x == 3:
                    if len(open_3s_black) > 0:
                        pos = random.choice(open_3s_black)
                        return pos
                    if len(half_open_3s_black) > 0:
                        pos = random.choice(half_open_3s_black)
                        return chose_empty_from_half_open(board, pos)
                if x == 2:
                    if len(open_2s_black) > 0:
                        pos = random.choice(open_2s_black)
                        return pos
                    if len(half_open_2s_black) > 0:
                        pos = random.choice(half_open_2s_black)
                        return chose_empty_from_half_open(board, pos)
            if color == 1:
                if x == 4:
                    if len(open_4s_black) > 0:
                        start_pos_1, end_pos_1, start_pos_2, end_pos_2 = random.choice(open_4s_black)
                        return np.array([start_pos_1, end_pos_1])
                    if len(half_open_4s_black) > 0:
                        pos = random.choice(half_open_4s_black)
                        return chose_empty_from_half_open(board, pos)
                if x == 3:
                    if len(open_3s_black) > 0:
                        pos = random.choice(open_3s_black)
                        return pos
                    if len(half_open_3s_black) > 0:
                        pos = random.choice(half_open_3s_black)
                        return chose_empty_from_half_open(board, pos)
                if x == 2:
                    if len(open_2s_black) > 0:
                        pos = random.choice(open_2s_black)
                        return pos
                    if len(half_open_2s_black) > 0:
                        pos = random.choice(half_open_2s_black)
                        return chose_empty_from_half_open(board, pos)
            else:
                if x == 4:
                    if len(open_4s_white) > 0:
                        start_pos_1, end_pos_1, start_pos_2, end_pos_2 = random.choice(open_4s_white)
                        return np.array([start_pos_1, end_pos_1])
                    if len(half_open_4s_white) > 0:
                        pos = random.choice(half_open_4s_white)
                        return chose_empty_from_half_open(board, pos)
                if x == 3:
                    if len(open_3s_white) > 0:
                        pos = random.choice(open_3s_white)
                        return pos
                    if len(half_open_3s_white) > 0:
                        pos = random.choice(half_open_3s_white)
                        return chose_empty_from_half_open(board, pos)
                if x == 2:
                    if len(open_2s_white) > 0:
                        pos = random.choice(open_2s_white)
                        return pos
                    if len(half_open_2s_white) > 0:
                        pos = random.choice(half_open_2s_white)
                        return chose_empty_from_half_open(board, pos)
        
        valid_moves = np.argwhere(board == 0)
        opponent_moves = np.argwhere(board == 3 - color)
        opponent_moves = np.random.permutation(opponent_moves)
        for stone_pos in opponent_moves:
            stone_pos_row = stone_pos[0]
            stone_pos_col = stone_pos[1]
            for i in range(max(0, stone_pos_row-2), min(15, stone_pos_row+3)):
                for j in range(max(0, stone_pos_col-2), min(15, stone_pos_col+3)):
                    if np.any(np.all(valid_moves == [i, j], axis=1)): 
                        return np.array([i, j])
        
        if board[7, 7] == 0:
            return (7, 7)
        valid_moves = np.argwhere(board == 0)
        for dist_to_middle in range(1,7):
            tl = [7-dist_to_middle, 7-dist_to_middle]
            top = [7-dist_to_middle, 7]
            tr = [7-dist_to_middle, 7+dist_to_middle]
            right = [7, 7+dist_to_middle]
            br = [7+dist_to_middle, 7+dist_to_middle]
            bottom = [7+dist_to_middle, 7]
            bl = [7+dist_to_middle, 7-dist_to_middle]
            left = [7, 7-dist_to_middle]
            circle_around_middle = [tl, top, tr, right, br, bottom, bl, left]
            while True:
                if len(circle_around_middle) == 0:
                    break
                rdn_from_circle = random.choice(circle_around_middle)
                if rdn_from_circle in valid_moves:
                    return rdn_from_circle
                else:
                    circle_around_middle.remove(rdn_from_circle)
        tl = [0,0]
        tr = [0,14]
        bl = [14,0]
        br = [14,14]
        for valid_move in valid_moves:
            if any(np.array_equal(valid_move, edge) for edge in [tl, tr, bl, br]):
                continue
            else:
                return valid_move
        
        for edge_pos in [tl, tr, bl, br]:
            if board[edge_pos[0], edge_pos[1]] == 0:
                return edge_pos