import time
from agent import Agent
import numpy as np
import eval_numba as eval
import random
from tqdm import tqdm


class AgentMinimax(Agent):
    def __init__(self, depth=4):
        self.depth = depth
        self.count = 0


    def get_valid_moves(self, board):
        return np.argwhere(board == 0)
    

    def append_ordered_remove_all_valid_moves(self, ordered_moves, start_row, start_col, end_row, end_col, all_valid_moves_list):
        '''
        Appends the start and end position of a move to the ordered_moves list and removes them from the all_valid_moves_list

        Args:
            ordered_moves (list): List of ordered moves
            start_row (int): start row of the move
            start_col (int): start column of the move
            end_row (int): end row of the move
            end_col (int): end column of the move
            all_valid_moves_list (list): List of all valid moves

        Returns:
            list, list: updated ordered_moves list, updated all_valid_moves_list list
        '''
        ordered_moves.append([start_row, start_col])
        ordered_moves.append([end_row, end_col])
        try:
            all_valid_moves_list.remove((start_row, start_col))
        except ValueError:
            try:
                all_valid_moves_list.remove((end_row, end_col))
            except ValueError:
                return ordered_moves, all_valid_moves_list
            return ordered_moves, all_valid_moves_list
        return ordered_moves, all_valid_moves_list

    
    def get_valid_moves_ordered(self, board, color):
        all_valid_moves = self.get_valid_moves(board)
        all_valid_moves_list = list(map(tuple, all_valid_moves))
        ordered_moves = []
        half_open_xs_black, open_xs_black, half_open_xs_white, open_xs_white = eval.x_in_row_finder(board)
        default_value = np.array([-1, -1, -1, -1], dtype=np.int8)
        if color == 1:
            own_half_open_xs = half_open_xs_black
            own_open_xs = open_xs_black
            opponent_open_xs = open_xs_white
            opponent_half_open_xs = half_open_xs_white
        else:
            own_half_open_xs = half_open_xs_white
            own_open_xs = open_xs_white
            opponent_open_xs = open_xs_black
            opponent_half_open_xs = half_open_xs_black
        
        own_open_4s = own_open_xs[2][~np.all(own_open_xs[2] == default_value, axis=1)]
        own_half_open_4s = own_half_open_xs[2][~np.all(own_half_open_xs[2] == default_value, axis=1)]
        own_open_3s = own_open_xs[1][~np.all(own_open_xs[1] == default_value, axis=1)]
        own_half_open_3s = own_half_open_xs[1][~np.all(own_half_open_xs[1] == default_value, axis=1)]
        own_open_2s = own_open_xs[0][~np.all(own_open_xs[0] == default_value, axis=1)]
        own_half_open_2s = own_half_open_xs[0][~np.all(own_half_open_xs[0] == default_value, axis=1)]
        opponent_open_4s = opponent_open_xs[2][~np.all(opponent_open_xs[2] == default_value, axis=1)]
        opponent_half_open_4s = opponent_half_open_xs[2][~np.all(opponent_half_open_xs[2] == default_value, axis=1)]
        opponent_open_3s = opponent_open_xs[1][~np.all(opponent_open_xs[1] == default_value, axis=1)]
        opponent_half_open_3s = opponent_half_open_xs[1][~np.all(opponent_half_open_xs[1] == default_value, axis=1)]
        opponent_open_2s = opponent_open_xs[0][~np.all(opponent_open_xs[0] == default_value, axis=1)]
        opponent_half_open_2s = opponent_half_open_xs[0][~np.all(opponent_half_open_xs[0] == default_value, axis=1)]
        for i in range(len(own_open_4s)):
            start_row, start_col, end_row, end_col = own_open_4s[i]
            ordered_moves, all_valid_moves_list = self.append_ordered_remove_all_valid_moves(ordered_moves, start_row, start_col, end_row, end_col, all_valid_moves_list)
        for i in range(len(own_half_open_4s)):
            open_side = eval.chose_empty_from_half_open(board, own_half_open_4s[i])
            ordered_moves.append(open_side)
            try:
                all_valid_moves_list.remove((open_side[0], open_side[1]))
            except ValueError:
                continue
        for i in range(len(opponent_open_4s)):
            start_row, start_col, end_row, end_col = opponent_open_4s[i]
            ordered_moves, all_valid_moves_list = self.append_ordered_remove_all_valid_moves(ordered_moves, start_row, start_col, end_row, end_col, all_valid_moves_list)
        for i in range(len(opponent_half_open_4s)):
            open_side = eval.chose_empty_from_half_open(board, opponent_half_open_4s[i])
            ordered_moves.append(open_side)
            try:
                all_valid_moves_list.remove((open_side[0], open_side[1]))
            except ValueError:
                continue
        for i in range(len(own_open_3s)):
            start_row, start_col, end_row, end_col = own_open_3s[i]
            ordered_moves, all_valid_moves_list = self.append_ordered_remove_all_valid_moves(ordered_moves, start_row, start_col, end_row, end_col, all_valid_moves_list)
        for i in range(len(opponent_open_3s)):
            start_row, start_col, end_row, end_col = opponent_open_3s[i]
            ordered_moves, all_valid_moves_list = self.append_ordered_remove_all_valid_moves(ordered_moves, start_row, start_col, end_row, end_col, all_valid_moves_list)
        for i in range(len(own_half_open_3s)):
            open_side = eval.chose_empty_from_half_open(board, own_half_open_3s[i])
            ordered_moves.append(open_side)
            try:
                all_valid_moves_list.remove((open_side[0], open_side[1]))
            except ValueError:
                continue
        for i in range(len(opponent_half_open_3s)):
            open_side = eval.chose_empty_from_half_open(board, opponent_half_open_3s[i])
            ordered_moves.append(open_side)
            try:
                all_valid_moves_list.remove((open_side[0], open_side[1]))
            except ValueError:
                continue
        for i in range(len(own_open_2s)):
            start_row, start_col, end_row, end_col = own_open_2s[i]
            ordered_moves, all_valid_moves_list = self.append_ordered_remove_all_valid_moves(ordered_moves, start_row, start_col, end_row, end_col, all_valid_moves_list)
        for i in range(len(opponent_open_2s)):
            start_row, start_col, end_row, end_col = opponent_open_2s[i]
            ordered_moves, all_valid_moves_list = self.append_ordered_remove_all_valid_moves(ordered_moves, start_row, start_col, end_row, end_col, all_valid_moves_list)
        for i in range(len(own_half_open_2s)):
            open_side = eval.chose_empty_from_half_open(board, own_half_open_2s[i])
            ordered_moves.append(open_side)
            try:
                all_valid_moves_list.remove((open_side[0], open_side[1]))
            except ValueError:
                continue
        for i in range(len(opponent_half_open_2s)):
            opponent_half_open_2s[i]
            open_side = eval.chose_empty_from_half_open(board, opponent_half_open_2s[i])
            ordered_moves.append(open_side)
            try:
                all_valid_moves_list.remove((open_side[0], open_side[1]))
            except ValueError:
                continue

        played_positions = np.argwhere(board != 0)
        for i in range(len(played_positions)):
            row, col = played_positions[i]
            for i in range(max(0, row-1), min(15, row+2)):
                for j in range(max(0, col-1), min(15, col+2)):
                    if board[i, j] == 0:
                        ordered_moves.append([i, j])
                        try:
                            all_valid_moves_list.remove((i, j))
                        except ValueError:
                            continue

        return ordered_moves
    
    
    def minimax(self, board, depth, alpha, beta, maximizing_player):
        if depth == 0 or eval.is_won(board) != 0:
            self.count += 1
            score = eval.eval(board)
            return score
        
        color = 1 if maximizing_player else 2
        valid_moves = self.get_valid_moves_ordered(board, color)
        
        if maximizing_player:
            max_eval = float('-inf')
            for move in valid_moves:
                board[move[0], move[1]] = 1
                score = self.minimax(board, depth - 1, alpha, beta, False)
                board[move[0], move[1]] = 0
                max_eval = max(max_eval, score)
                alpha = max(alpha, score)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in valid_moves:
                board[move[0], move[1]] = 2
                score = self.minimax(board, depth - 1, alpha, beta, True)
                board[move[0], move[1]] = 0
                min_eval = min(min_eval, score)
                beta = min(beta, score)
                if beta <= alpha:
                    break
            return min_eval

    def place_stone(self, board, color):
        color = color
        board = np.copy(board)
        if color == 1:
            maximizing_player = True
        else:
            maximizing_player = False
        best_action = None
        best_value = float('-inf') if color == 1 else float('inf')
        start = time.time()
        valid_moves = self.get_valid_moves_ordered(board, color)
        if len(valid_moves) == 0:
            valid_moves = [(random.randint(6,8), random.randint(6,8))]
        for i in tqdm(range(len(valid_moves))):
            move = valid_moves[i]
            board[move[0], move[1]] = color
            score = self.minimax(board, self.depth-1, float('-inf'), float('inf'), not maximizing_player)
            board[move[0], move[1]] = 0
            if (color == 1 and score > best_value) or (color == 2 and score < best_value):
                best_value = score
                best_action = move
        print(f"Number of leafs visited: {self.count}")
        self.count = 0
        print(f"Time taken: {time.time() - start}")
        return tuple(best_action)