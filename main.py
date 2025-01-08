import numpy as np
import eval_numba as eval
from agent_human import AgentHuman
from agent_random import AgentRandom
from agent_follow import AgentFollowOpponent
from agent_eager import AgentEager
from agent_minimax import AgentMinimax
from agent_nnue import AgentNNUE
from agent_classic import AgentClassic
import time

def print_board(board):
    board_str = "     0  1  2  3  4  5  6  7  8  9  10 11 12 13 14\n"
    board_str += "   ----------------------------------------------\n"
    for row in range(15):
        if row < 10:
            board_str += f" {row}| "
        else:
            board_str += f"{row}| "
        for col in range(15-1):
                board_str += f" {board[row, col]} "
        board_str += f" {board[row, 14]}"
        if row < 10:
            board_str += f" | {row}\n"
        else:
            board_str += f" |{row}\n"
    board_str += "   ----------------------------------------------\n"
    board_str += "     0  1  2  3  4  5  6  7  8  9 10 11 12 13 14\n"
    print(board_str)


def input_handler():
    '''
    Handles the necessary input of the user to set up the game and what agents are playing
    '''
    while True:
        agent_black_name = input('Select the agent that will play for Black (1) [human, random, follow, eager, minimax, nnue, classic]: ')
        if agent_black_name not in ['human', 'random', 'follow', 'eager', 'minimax', 'nnue', 'classic']:
            print(f'You entered {agent_black_name}, expected input: human, random, follow, eager, minimax, nnue or classic.')
            continue
        if agent_black_name == 'human':
            agent_black = AgentHuman()
        if agent_black_name == 'random':
            agent_black = AgentRandom()
        if agent_black_name == 'follow':
            agent_black = AgentFollowOpponent()
        if agent_black_name == 'eager':
            agent_black = AgentEager()
        if agent_black_name == 'minimax':
            agent_black = AgentMinimax()
        if agent_black_name == 'nnue':
            agent_black = AgentNNUE()
        if agent_black_name == 'classic':
            agent_black = AgentClassic()
        print(f'{agent_black_name} is playing Black.')
        break
        
    while True:
        agent_white_name = input('Select the agent that will play for White (2) [human, random, follow, eager, minimax, nnue, classic]: ')
        if agent_white_name not in ['human', 'random', 'follow', 'eager', 'minimax', 'nnue', 'classic']:
            print(f'You entered {agent_white_name}, expected input: human, random, follow, eager, minimax, nnue or classic.')
            continue
        if agent_white_name == 'human':
            agent_white = AgentHuman()
        if agent_white_name == 'random':
            agent_white = AgentRandom()
        if agent_white_name == 'follow':
            agent_white = AgentFollowOpponent()
        if agent_white_name == 'eager':
            agent_white = AgentEager()
        if agent_white_name == 'minimax':
            agent_white = AgentMinimax()
        if agent_white_name == 'nnue':
            agent_white = AgentNNUE()
        if agent_white_name == 'classic':
            agent_white = AgentClassic()
        print(f'{agent_white_name} is playing White.')
        break
    return agent_black, agent_white


def place_stone(board, color, pos):
    '''
    Places stone onto the board, expects preceding check for a valid move

    Args:
        color (char): Either 1 for black stone or -1 for white stone
        pos (tuple): Position at where the stone should be placed at on the board
    '''
    row = pos[0]
    col = pos[1]
    board[row, col] = color
    return board


def main(board = np.zeros((15,15), dtype=np.int8), curr_color = 1):
    board = board
    curr_color = curr_color
    number_moves = 0
    agent_black, agent_white = input_handler()
    while True:
        if curr_color == 1:
            print("Black's turn!")
            pos = agent_black.place_stone(board, 1)
        else:
            print("White's turn!")
            pos = agent_white.place_stone(board, 2)

        board = place_stone(board, curr_color, pos)
        number_moves += 1

        print_board(board)
        start = time.perf_counter()
        board_eval = eval.eval(board)
        print(f"Time for eval: {time.perf_counter() - start} seconds")
        print(f"Board evaluation: {board_eval}")
        print("############")
        number_valid_moves = len(eval.get_valid_moves(board))
        if board_eval == 10_000:
            print(f'Black won in {number_moves} moves!')
            break
        if board_eval == -10_000:
            print(f'White won in {number_moves} moves!')
            break
        if number_valid_moves == 0:
            print(f'Draw after {number_moves} moves!')
            break
        if curr_color == 1:
            curr_color = 2
        else:
            curr_color = 1

if __name__ == "__main__":
    test_board = np.zeros((15, 15), dtype=np.int8)
    eval.eval(test_board)
    main(board=test_board, curr_color=1)
    