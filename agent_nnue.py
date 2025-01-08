import time
from agent_minimax import AgentMinimax
import numpy as np
import eval_numba as eval
import torch
from nnue import NNUEGomoku
import random
from tqdm import tqdm

class AgentNNUE(AgentMinimax):
    def __init__(self, model_path='nnue_gomoku_model.pth', device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), depth=4):
        super().__init__(depth=depth)
        self.count = 0
        self.model = NNUEGomoku()
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()


    def eval(self, board):
        board_t = torch.from_numpy(board)
        player1_board = (board_t == 1).to(torch.float32).flatten()
        player2_board = (board_t == 2).to(torch.float32).flatten()

        start = time.perf_counter_ns()
        with torch.no_grad():
            score = self.model(player1_board.unsqueeze(0), player2_board.unsqueeze(0)).item()
        return score
    

    def minimax(self, board, depth, alpha, beta, maximizing_player):
        is_won = eval.is_won(board)
        if is_won == 1:
            self.count += 1
            return 10_000
        if is_won == 2:
            self.count += 1
            return -10_000
        if depth == 0:
            self.count += 1
            score = self.eval(board)
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
        valid_moves = super().get_valid_moves_ordered(board, color)
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
