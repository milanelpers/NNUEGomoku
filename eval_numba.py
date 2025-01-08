import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numba import njit
import time


BOARD_SIZE = 15


@njit
def get_valid_moves(board):
    '''
    Returns the valid moves for the current board state.

    Args:
        board (ndarray): The current board state.

    Returns:
        ndarray: Array of tuples containing the valid moves.
    '''
    return np.argwhere(board == 0)


@njit
def chose_empty_from_half_open(board, pos_of_half_open):
    '''
    Chooses the empty position from a given tuple containing the first element of the half-open x-in-row of the opponent

    Args:
        board (ndarray): The board where the stone will be placed.
        pos_of_half_open (ndarray): (4,) shaped array containing the pos of the first element of the half-open x-in-row of the opponent or the board edge and the empty pos

    Returns:
        ndarray: The empty position the agent will play, shaped (2,)
    '''
    pos_1_start, pos_1_end, pos_2_start, pos_2_end = pos_of_half_open
    if board[pos_1_start, pos_1_end] == 0:
        return pos_of_half_open[0:2]
    else:
        return pos_of_half_open[2:4]


@njit
def area_around_pos(board, row, col):
    '''
    Returns the slice for the area around a position on the board.

    Args:
        board (ndarray): The board
        row (int): The row of the position
        col (int): The column of the position

    Returns:
        board_area (ndarray): The area of the board around the given position // Tuple of the row-slice and the column-slice for the area around the given position
    '''
    if row == 0 and col == 0:
        row_slice = slice(0, 2)
        col_slice = slice(0, 2)
    if row == 0 and col == BOARD_SIZE-1:
        row_slice = slice(0, 2)
        col_slice = slice(13, 15)
    if row == BOARD_SIZE-1 and col == 0:
        row_slice = slice(13, 15)
        col_slice = slice(0, 2)
    if row == BOARD_SIZE-1 and col == BOARD_SIZE-1:
        row_slice = slice(13, 15)
        col_slice = slice(13, 15)
    if row == 0 and col != 0 and col != BOARD_SIZE-1:
        row_slice = slice(0, 2)
        col_slice = slice(col-1, col+2)
    if row == BOARD_SIZE-1 and col != 0 and col != BOARD_SIZE-1:
        row_slice = slice(13, 15)
        col_slice = slice(col-1, col+2)
    if col == 0 and row != 0 and row != BOARD_SIZE-1:
        row_slice = slice(row-1, row+2)
        col_slice = slice(0, 2)
    if col == BOARD_SIZE-1 and row != 0 and row != BOARD_SIZE-1:
        row_slice = slice(row-1, row+2)
        col_slice = slice(13, 15)
    if row != 0 and row != BOARD_SIZE-1 and col != 0 and col != BOARD_SIZE-1:
        row_slice = slice(row-1, row+2)
        col_slice = slice(col-1, col+2)
    board_area = board[row_slice, col_slice]
    return board_area


@njit
def free_pos(board):
    '''
    Calculates the score of the board concerning the free positions around stones.
    Each stone adds number_free_positions_around_stone to the score (+9 for positions in the middle for black, -9 for white; +-3 for corners)

    Args:
        board (ndarray): The board to evaluate

    Returns:
        score (int): The score from the free positions around the stones
    '''
    score = 0
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if board[row, col] == 1:
                score += np.sum(area_around_pos(board, row, col) == 0)
            if board[row, col] == 2:
                score -= np.sum(area_around_pos(board, row, col) == 0)
    return score


@njit
def can_skip(slice):
    '''
    Checks whether or not we can skip evaluating the slice for a potential x-in-a-row:
        - Empty slices
        - Slices with only one stone of either black or white player
        - Mixed slices, e.g. 1 2 0 0 0, 2 1 2 1 2 (not valuable since it cannot lead to a 5-in-row and we would just give the oponent time where they can make good moves while we do not)

    Args:
        slice (ndarray): A 6-long slice from the board
    Returns:
        Boolean: True if the slice can get skipped, False otherwise
    '''
    if np.all(slice == 0):
        return True
    if np.sum(slice) == 1 or (np.sum(slice) == 2 and np.all(slice != 1)):
        return True
    if np.any(slice == 1) and np.any(slice == 2):
        return True
    return False


@njit
def is_won(board):
    '''
    Checks the board for 5-in-a-row for either player.

    Args:
        board (ndarray): The board

    Returns:
        int: 1 if black player won, 2 if white player won, 0 if the game isn't won yet.
    '''
    slides_h = sliding_window_view(board, window_shape=5, axis=1)
    slides_v = sliding_window_view(board.T, window_shape=5, axis=1)
    for line in range(BOARD_SIZE):
        for slide_idx in range(slides_h.shape[1]):
            curr_slide_h = slides_h[line, slide_idx]
            curr_slide_v = slides_v[line, slide_idx]
            if np.all(curr_slide_h == 1) or np.all(curr_slide_v == 1):
                return 1
            if np.all(curr_slide_h == 2) or np.all(curr_slide_v == 2):
                return 2
    for offset in range(-10,11):
        diag = np.diag(board, k=offset)
        slides_d = sliding_window_view(diag, window_shape=5)
        anti_diag = np.diag(np.fliplr(board), k=offset)
        slides_ad = sliding_window_view(anti_diag, window_shape=5)
        for slide_idx in range(slides_d.shape[0]):
            curr_slide_d = slides_d[slide_idx]
            curr_slide_ad = slides_ad[slide_idx]
            if np.all(curr_slide_d == 1) or np.all(curr_slide_ad == 1):
                return 1
            if np.all(curr_slide_d == 2) or np.all(curr_slide_ad == 2):
                return 2
    return 0


@njit
def pos_tuple_h(line, slide_idx):
    '''
    Calculates the start and end position of a 5-wide horizontal slide and returns them via a tuple.

    Args:
        line (int): The current row
        slide_idx (int): The "column-offset" of the current row where the 5-wide slide starts

    Returns:
        tuple: Tuple of tuples storing the start and end position of a 5-wide slide.
    '''
    return ((line, slide_idx), (line, slide_idx+4))


@njit
def pos_tuple_v(line, slide_idx):
    '''
    Calculates the start and end position of a 5-wide vertical slide and returns them via a tuple.

    Args:
        line (int): The current column
        slide_idx (int): The "column-offset" of the current column where the 5-wide slide starts

    Returns:
        tuple: Tuple of tuples storing the start and end position of a 5-wide slide.
    '''
    return ((slide_idx, line), (slide_idx+4, line))


@njit
def pos_tuple_d(offset, slide_idx):
    '''
    Calculates the start and end position of a 5-wide diagonal slide and returns them via a tuple.

    Args:
        offset (int): The current offset-diagonal. If 0, it's the main diagonal
        slide_idx (int): The "column-offset" of the current diagonal where the 5-wide slide starts

    Returns:
        tuple: Tuple of tuples storing the start and end position of a 5-wide slide.
    '''
    if offset < 0:
        return ((slide_idx+abs(offset), slide_idx), (slide_idx+abs(offset)+4, slide_idx+4))
    else:
        return ((slide_idx, slide_idx+offset), (slide_idx+4, slide_idx+offset+4))


@njit
def pos_tuple_ad(offset, slide_idx):
    '''
    Calculates the start and end position of a 5-wide anti-diagonal slide and returns them via a tuple.

    Args:
        offset (int): The current offset-anti-diagonal. If 0, it's the main diagonal
        slide_idx (int): The "column-offset" of the current anti-diagonal where the 5-wide slide starts

    Returns:
        tuple: Tuple of tuples storing the start and end position of a 5-wide slide.
    '''
    if offset < 0:
        return ((slide_idx+abs(offset), 14-slide_idx), (slide_idx+abs(offset)+4, 14-slide_idx-4))
    else:
        return ((slide_idx, 14-offset-slide_idx), (slide_idx+4, 14-offset-slide_idx-4))
    

@njit
def create_empty_arrays():
    '''
    Creates empty arrays for open and half-open xs-in-a row.
    
    Returns:
        half_open_xs_black (ndarray): (3, 200, 4) shaped ndarray for half-open xs-in-row for black
        open_xs_black (ndarray): (3, 200, 4) shaped ndarray for open xs-in-row for black
        half_open_xs_white (ndarray): (3, 200, 4) shaped ndarray for half-open xs-in-row for white
        open_xs_white (ndarray): (3, 200, 4) shaped ndarray for open xs-in-row for white
    '''
    half_open_xs_black = np.full(shape=(3, 200, 4), fill_value=-1, dtype=np.int8)
    open_xs_black = np.full(shape=(3, 200, 4), fill_value=-1, dtype=np.int8)
    half_open_xs_white = np.full(shape=(3, 200, 4), fill_value=-1, dtype=np.int8)
    open_xs_white = np.full(shape=(3, 200, 4), fill_value=-1, dtype=np.int8)
    return half_open_xs_black, open_xs_black, half_open_xs_white, open_xs_white


@njit
def store_pos(array, pos):
    '''
    Stores the position in the structured array for x at the first empty spot.

    Args:
        array (ndarray): The (200, 4) shaped array storing start and end positions of xs-in-a-row
        pos (ndarray): The start and end positions of an x-in-a-row, shaped (4,)

    Returns:
        array (ndarray): The (200, 4) shaped updated array with the position stored
    '''
    default_value = np.array([-1, -1, -1, -1], dtype=np.int8)
    if np.all(pos == default_value):
        return array
    where_empty = array == default_value
    free_pos_offset = np.argmax(where_empty, axis=0)[0]
    array[free_pos_offset] = pos
    return array


@njit
def merge_arrays(source_array, dest_array):
    '''
    Merges two (3, 200, 4) shaped ndarrays by iterating over the source array and storing the positions in the destination array.

    Args:
        source_array (ndarray): The array to iterate over, shaped (3, 200, 4)
        dest_arrays (ndarray): The array to store the positions in, shaped (3, 200, 4)

    Returns:
        dest_dict (dict): The destination array with the positions stored
    '''
    default_value = np.array([-1, -1, -1, -1], dtype=np.int8)
    for x in range(2,5):
        source_arr = source_array[x-2]
        dest_arr = dest_array[x-2]
        free_pos_offset = np.argmax(dest_arr == default_value, axis=0)[0]
        positions_indices = source_arr != default_value
        if np.all(positions_indices == False):
            continue
        las_pos_idx = positions_indices.nonzero()[0][-1]
        positions = source_arr[slice(0, las_pos_idx+1)]
        indices_new_positions = slice(free_pos_offset, free_pos_offset+len(positions))
        dest_arr[indices_new_positions] = positions
        dest_array[x-2] = dest_arr
    return dest_array


@njit
def process_lines(board, slides, orientation):
    '''
    Iterates over the lines (either rows or columns) of the board and checks for x-in-a-row.
    Stores the positions of the x-in-a-row in the respective arrays.

    Args:
        board (ndarray): The board
        slides (ndarray): The 5-wide slides of the board, shaped (15, 11, 5)
        orientation (str): The orientation of the slides, so either horizontal or vertical

    Returns:
        local_half_open_xs_black (ndarray): Dictionary for half-open xs-in-row for black
        local_open_xs_black (ndarray): Dictionary for open xs-in-row for black
        local_half_open_xs_white (ndarray): Dictionary for half-open xs-in-row for white
        local_open_xs_white (ndarray): Dictionary for open xs-in-row for white
    '''
    local_half_open_xs_black, local_open_xs_black, local_half_open_xs_white, local_open_xs_white = create_empty_arrays()
    for line in range(BOARD_SIZE):
        if orientation == 'horizontal' and np.sum(board[line]) == 0:
            continue
        if orientation == 'vertical' and np.sum(board[:, line]) == 0:
            continue
        slides_curr_line = slides[line]
        if orientation == 'horizontal':
            full_line = board[line, :]
        else:
            full_line = board[:, line]
        slide_idx = 0
        while slide_idx < len(slides_curr_line):
            curr_slide = slides_curr_line[slide_idx]
            if can_skip(curr_slide):
                slide_idx += 1
                continue
            if slide_idx < len(slides_curr_line)-1:
                next_slide = slides_curr_line[slide_idx+1]
                wait_for_next_slide = False
                for x in range(2,5):
                    if curr_slide[-1] != 0 and curr_slide[-1] == next_slide[-1] and np.all(curr_slide[-x:] == curr_slide[-1]):
                        wait_for_next_slide = True
                if wait_for_next_slide:
                    slide_idx += 1
                    continue
            x = 4
            while x > 1:
                for slide_start in range(5-x+1):
                    curr_subslide = curr_slide[slide_start:slide_start+x]
                    if orientation == 'horizontal':
                        pos_tuple = pos_tuple_h(line, slide_idx)
                    else:
                        pos_tuple = pos_tuple_v(line, slide_idx)
                    start_pos_row = pos_tuple[0][0]
                    start_pos_col = pos_tuple[0][1]
                    end_pos_row = pos_tuple[1][0]
                    end_pos_col = pos_tuple[1][1]
                    has_changed = False
                    if np.all(curr_subslide == 1):
                        if slide_start == 0 and (slide_idx==0 or full_line[slide_idx-1] == 2):
                            if orientation == 'horizontal':
                                pos = np.array([start_pos_row, start_pos_col, start_pos_row, start_pos_col+x])
                                local_half_open_xs_black[x-2] = store_pos(local_half_open_xs_black[x-2], pos)
                            else:
                                pos = np.array([start_pos_row, start_pos_col, start_pos_row+x, start_pos_col])
                                local_half_open_xs_black[x-2] = store_pos(local_half_open_xs_black[x-2], pos)
                        elif slide_start == 5-x and (slide_idx+slide_start+x >= len(full_line) or full_line[slide_idx+slide_start+x] == 2):
                            if orientation == 'horizontal':
                                pos = np.array([end_pos_row, end_pos_col-x, end_pos_row, end_pos_col])
                                local_half_open_xs_black[x-2] = store_pos(local_half_open_xs_black[x-2], pos)
                            else:
                                pos = np.array([end_pos_row-x, end_pos_col, end_pos_row, end_pos_col])
                                local_half_open_xs_black[x-2] = store_pos(local_half_open_xs_black[x-2], pos)
                        else:
                            if orientation == 'horizontal':
                                pos = np.array([start_pos_row, start_pos_col+slide_start-1, start_pos_row, start_pos_col+slide_start+x])
                                local_open_xs_black[x-2] = store_pos(local_open_xs_black[x-2], pos)
                            else:
                                pos = np.array([start_pos_row+slide_start-1, start_pos_col, start_pos_row+slide_start+x, start_pos_col])
                                local_open_xs_black[x-2] = store_pos(local_open_xs_black[x-2], pos)
                        has_changed = True
                        slide_idx += slide_start+x
                        x = 1
                        break
                    if np.all(curr_subslide == 2):
                        if slide_start == 0 and (slide_idx==0 or full_line[slide_idx-1] == 1):
                            if orientation == 'horizontal':
                                pos = np.array([start_pos_row, start_pos_col, start_pos_row, start_pos_col+x])
                                local_half_open_xs_white[x-2] = store_pos(local_half_open_xs_white[x-2], pos)
                            else:
                                pos = np.array([start_pos_row, start_pos_col, start_pos_row+x, start_pos_col])
                                local_half_open_xs_white[x-2] = store_pos(local_half_open_xs_white[x-2], pos)
                        elif slide_start == 5-x and (slide_idx+slide_start+x >= len(full_line) or full_line[slide_idx+slide_start+x] == 1):
                            if orientation == 'horizontal':
                                pos = np.array([end_pos_row, end_pos_col-x, end_pos_row, end_pos_col])
                                local_half_open_xs_white[x-2] = store_pos(local_half_open_xs_white[x-2], pos)
                            else:
                                pos = np.array([end_pos_row-x, end_pos_col, end_pos_row, end_pos_col])
                                local_half_open_xs_white[x-2] = store_pos(local_half_open_xs_white[x-2], pos)
                        else:
                            if orientation == 'horizontal':
                                pos = np.array([start_pos_row, start_pos_col+slide_start-1, start_pos_row, start_pos_col+slide_start+x])
                                local_open_xs_white[x-2] = store_pos(local_open_xs_white[x-2], pos)
                            else:
                                pos = np.array([start_pos_row+slide_start-1, start_pos_col, start_pos_row+slide_start+x, start_pos_col])
                                local_open_xs_white[x-2] = store_pos(local_open_xs_white[x-2], pos)
                        has_changed = True
                        slide_idx += slide_start+x
                        x = 1
                        break
                x -= 1
            if has_changed:
                continue
            else:
                slide_idx += 1
    return local_half_open_xs_black, local_open_xs_black, local_half_open_xs_white, local_open_xs_white


@njit
def process_diags(board, orientation):
    '''
    Iterates over the diagonals of the board and checks for x-in-a-row.
    Stores the positions of the x-in-a-row in the respective arrays.

    Args:
        board (ndarray): The board
        orientation (str): The orientation of the slides, so either diagonal or anti-diagonal

    Returns:
        local_half_open_xs_black (ndarray): Array for half-open xs-in-row for black
        local_open_xs_black (ndarray): Array for open xs-in-row for black
        local_half_open_xs_white (ndarray): Array for half-open xs-in-row for white
        local_open_xs_white (ndarray): Array for open xs-in-row for white
    '''
    local_half_open_xs_black, local_open_xs_black, local_half_open_xs_white, local_open_xs_white = create_empty_arrays()
    for offset in range(-10, 11):
        if orientation == 'diagonal':
            diag = np.diag(board, k=offset)
        else:
            diag = np.diag(np.fliplr(board), k=offset)
        slides = sliding_window_view(diag, window_shape=5)
        if np.sum(diag) == 0:
            continue
        slide_idx = 0
        while slide_idx < len(slides):
            curr_slide = slides[slide_idx]
            if can_skip(curr_slide):
                slide_idx += 1
                continue
            if slide_idx < len(slides)-1:
                next_slide = slides[slide_idx+1]
                wait_for_next_slide = False
                for x in range(2,5):
                    if curr_slide[-1] != 0 and curr_slide[-1] == next_slide[-1] and np.all(curr_slide[-x:] == curr_slide[-1]):
                        wait_for_next_slide = True
                if wait_for_next_slide:
                    slide_idx += 1
                    continue
            x = 4
            while x > 1:
                for slide_start in range(5-x+1):
                    curr_subslide = curr_slide[slide_start:slide_start+x]
                    if orientation == 'diagonal':
                        pos_tuple = pos_tuple_d(offset, slide_idx)
                    else:
                        pos_tuple = pos_tuple_ad(offset, slide_idx)
                    start_pos_row = pos_tuple[0][0]
                    start_pos_col = pos_tuple[0][1]
                    end_pos_row = pos_tuple[1][0]
                    end_pos_col = pos_tuple[1][1]
                    has_changed = False
                    if np.all(curr_subslide == 1):
                        if slide_start == 0 and (slide_idx==0 or diag[slide_idx-1] == 2):
                            if orientation == 'diagonal':
                                pos = np.array([start_pos_row, start_pos_col, start_pos_row+x, start_pos_col+x])
                                local_half_open_xs_black[x-2] = store_pos(local_half_open_xs_black[x-2], pos)
                            else:
                                pos = np.array([start_pos_row, start_pos_col, start_pos_row+x, start_pos_col-x])
                                local_half_open_xs_black[x-2] = store_pos(local_half_open_xs_black[x-2], pos)
                        elif slide_start == 5-x and (slide_idx+slide_start+x >= len(diag) or diag[slide_idx+slide_start+x] == 2):
                            if orientation == 'diagonal':
                                pos = np.array([end_pos_row-x, end_pos_col-x, end_pos_row, end_pos_col])
                                local_half_open_xs_black[x-2] = store_pos(local_half_open_xs_black[x-2], pos)
                            else:
                                pos = np.array([end_pos_row-x, end_pos_col+x, end_pos_row, end_pos_col])
                                local_half_open_xs_black[x-2] = store_pos(local_half_open_xs_black[x-2], pos)
                        else:
                            if orientation == 'diagonal':
                                pos = np.array([start_pos_row+slide_start-1, start_pos_col+slide_start-1, start_pos_row+slide_start+x, start_pos_col+slide_start+x])
                                local_open_xs_black[x-2] = store_pos(local_open_xs_black[x-2], pos)
                            else:
                                pos = np.array([start_pos_row+slide_start-1, start_pos_col-slide_start+1, start_pos_row+slide_start+x, start_pos_col-slide_start-x])
                                local_open_xs_black[x-2] = store_pos(local_open_xs_black[x-2], pos)
                        has_changed = True
                        slide_idx += slide_start+x
                        x = 1
                        break
                    if np.all(curr_subslide == 2):
                        if slide_start == 0 and (slide_idx==0 or diag[slide_idx-1] == 1):
                            if orientation == 'diagonal':
                                pos = np.array([start_pos_row, start_pos_col, start_pos_row+x, start_pos_col+x])
                                local_half_open_xs_white[x-2] = store_pos(local_half_open_xs_white[x-2], pos)
                            else:
                                pos = np.array([start_pos_row, start_pos_col, start_pos_row+x, start_pos_col-x])
                                local_half_open_xs_white[x-2] = store_pos(local_half_open_xs_white[x-2], pos)
                        elif slide_start == 5-x and (slide_idx+slide_start+x >= len(diag) or diag[slide_idx+slide_start+x] == 1):
                            if orientation == 'diagonal':
                                pos = np.array([end_pos_row-x, end_pos_col-x, end_pos_row, end_pos_col])
                                local_half_open_xs_white[x-2] = store_pos(local_half_open_xs_white[x-2], pos)
                            else:
                                pos = np.array([end_pos_row-x, end_pos_col+x, end_pos_row, end_pos_col])
                                local_half_open_xs_white[x-2] = store_pos(local_half_open_xs_white[x-2], pos)
                        else:
                            if orientation == 'diagonal':
                                pos = np.array([start_pos_row+slide_start-1, start_pos_col+slide_start-1, start_pos_row+slide_start+x, start_pos_col+slide_start+x])
                                local_open_xs_white[x-2] = store_pos(local_open_xs_white[x-2], pos)
                            else:
                                pos = np.array([start_pos_row+slide_start-1, start_pos_col-slide_start+1, start_pos_row+slide_start+x, start_pos_col-slide_start-x])
                                local_open_xs_white[x-2] = store_pos(local_open_xs_white[x-2], pos)
                        has_changed = True
                        slide_idx += slide_start+x
                        x = 1
                        break
                x -= 1
            if has_changed:
                continue
            else:
                slide_idx += 1
    return local_half_open_xs_black, local_open_xs_black, local_half_open_xs_white, local_open_xs_white

        
@njit
def x_in_row_finder(board):
    '''
    Iterates over the board and checks for all possible 5-position wide slices if there are xs in a row / col / diag / anti-diag.
    Differentiates between open xs and half open xs: open xs are where the xs is cornered by empty positions, half open xs are where only one end is still an open position.
    Stores the start and end positions of the slice containing xs-in-a-row.

    Slices can look like this:
    | 1 1 0 0 0 | -> if pos to the left of slice is the board edge or blocked by opponent -> half-open, otherwise open
    | 0 1 1 0 0 | -> open
    | 0 0 1 1 0 | -> open
    | 0 0 0 1 1 | -> if pos to the right of slice is the board edge or blocked by opponent -> half-open, otherwise open

    | 1 1 1 0 0 | -> if pos to the left of slice is the board edge or blocked by opponent -> half-open, otherwise open
    | 0 1 1 1 0 | -> open
    | 0 0 1 1 1 | -> if pos to the right of slice is the board edge or blocked by opponent -> half-open, otherwise open

    | 1 1 1 1 0 | -> if pos to the left of slice is the board edge or blocked by opponent -> half-open, otherwise open
    | 0 1 1 1 1 | -> if pos to the right of slice is the board edge or blocked by opponent -> half-open, otherwise open

    Args:
        board (ndarray): The current board state.

    Returns:
        open_xs_black (dict): Contains the open x-in-a-rows for the black player.
        half_open_xs_black (dict): Contains the half-open x-in-a-rows for the black player.
        open_xs_white (dict): Contains the open x-in-a-rows for the white player.
        half_open_xs_white (dict): Contains the half-open x-in-a-rows for the white player.

    Returns:
        half_open_xs_black, open_xs_black, half_open_xs_white, open_xs_white (dict): The updated dictionaries
    '''
    slides_h = sliding_window_view(board, window_shape=5, axis=1)
    slides_v = sliding_window_view(board.T, window_shape=5, axis=1)
    half_open_xs_black, open_xs_black, half_open_xs_white, open_xs_white = process_lines(board, slides_h, 'horizontal')
    half_open_xs_black_v, open_xs_black_v, half_open_xs_white_v, open_xs_white_v = process_lines(board, slides_v, 'vertical')
    half_open_xs_black = merge_arrays(half_open_xs_black_v, half_open_xs_black)
    open_xs_black = merge_arrays(open_xs_black_v, open_xs_black)
    half_open_xs_white = merge_arrays(half_open_xs_white_v, half_open_xs_white)
    open_xs_white = merge_arrays(open_xs_white_v, open_xs_white)

    half_open_xs_black_d, open_xs_black_d, half_open_xs_white_d, open_xs_white_d = process_diags(board, 'diagonal')
    half_open_xs_black = merge_arrays(half_open_xs_black_d, half_open_xs_black)
    open_xs_black = merge_arrays(open_xs_black_d, open_xs_black)
    half_open_xs_white = merge_arrays(half_open_xs_white_d, half_open_xs_white)
    open_xs_white = merge_arrays(open_xs_white_d, open_xs_white)
    half_open_xs_black_ad, open_xs_black_ad, half_open_xs_white_ad, open_xs_white_ad = process_diags(board, 'anti-diagonal')
    half_open_xs_black = merge_arrays(half_open_xs_black_ad, half_open_xs_black)
    open_xs_black = merge_arrays(open_xs_black_ad, open_xs_black)
    half_open_xs_white = merge_arrays(half_open_xs_white_ad, half_open_xs_white)
    open_xs_white = merge_arrays(open_xs_white_ad, open_xs_white)

    return half_open_xs_black, open_xs_black, half_open_xs_white, open_xs_white


@njit
def forks(open_xs_black, open_xs_white):
    '''
    Checks the board for forks:
    1 0 0 1 0 0 1
    0 1 0 1 0 1 0
    0 0 1 1 1 0 0
    1 1 1 X 1 1 1
    0 0 1 1 1 0 0
    0 1 0 1 0 1 0
    1 0 0 1 0 0 1

    Whenever 2 of open-threes are aligned like in the picture, if we place a stone in the spot with X, we automatically get 2 fours

    Args:
        open_xs_black (ndarray): (200, 4) shaped array, containing the open 3-in-a-rows for the black player.
        open_xs_white (ndarray): (200, 4) shaped array, containing the open 3-in-a-rows for the white player.

    Returns
        number_forks (ndarray): First element contains the number of forks found on the board for the black player, second contains the number for the white player
    '''
    number_forks = np.array([0, 0])
    for idx in range(open_xs_black.shape[0]):
        start_pos_1, end_pos_1, start_pos_2, end_pos_2 = open_xs_black[idx]
        if start_pos_1 == -1:
            break
        if start_pos_1 == start_pos_2 or start_pos_1 == end_pos_2 or end_pos_1 == start_pos_2 or end_pos_1 == end_pos_2:
            number_forks[0] += 1
    for idx in range(open_xs_white.shape[0]):
        start_pos_1, end_pos_1, start_pos_2, end_pos_2 = open_xs_white[idx]
        if start_pos_1 == -1:
            break
        if start_pos_1 == start_pos_2 or start_pos_1 == end_pos_2 or end_pos_1 == start_pos_2 or end_pos_1 == end_pos_2:
            number_forks[1] += 1
    return number_forks

@njit
def eval(board):
    '''
    Calculates the score of the current board. Black adds to the score, White removes from it
    - 0) No stones on board = 0 pts. Same on boards with 0 valid moves left.
    - 1) Each stone adds number_free_positions_around to the score (+8 for positions in the middle for black, -8 for white; +-3 for corners)
    - 2) x-in-a-row
        2-in-a-row and free positions next to it: 10pts, 20pts if both "sides" are free
        3-in-a-row and free positions next to it: 25pts, 50pts if both "sides" are free
        4-in-a-row and free positions next to it: 50pts, 100pts if both "sides" are free -> this is not defendable and the player will win
        if an x-in-a-row has no free positions next to it, it is worth 0pts, however
        5-in-a-row aka win-condition met: 10_000 pts for black, - 10_000 for white
    - 3) Forks -> 1000 pts

    Args:
        board (ndarray): The current board state.

    Returns:
        score (int): The score of the board
    '''
    score = 0
    
    is_won_result = is_won(board)
    if is_won_result == 1:
        return 10_000
    elif is_won_result == 2:
        return -10_000

    
    if len(np.argwhere(board)) <= 2:
        return score
    
    if len(get_valid_moves(board)) == 0:
        return 0
    
    half_open_xs_black, open_xs_black, half_open_xs_white, open_xs_white = x_in_row_finder(board)
    for x in range(2,5):
        idx_x = x-2
        for idx_elements in range(half_open_xs_black.shape[1]):
            start_pos_1, end_pos_1, start_pos_2, end_pos_2 = half_open_xs_black[idx_x][idx_elements]
            if start_pos_1 == -1:
                break
            else:
                score += (10**(x-1))
        for idx_elements in range(open_xs_black.shape[1]):
            start_pos_1, end_pos_1, start_pos_2, end_pos_2 = open_xs_black[idx_x][idx_elements]
            if start_pos_1 == -1:
                break
            else:
                score += 2*(10**(x-1))
        for idx_elements in range(half_open_xs_white.shape[1]):
            start_pos_1, end_pos_1, start_pos_2, end_pos_2 = half_open_xs_white[idx_x][idx_elements]
            if start_pos_1 == -1:
                break
            else:
                score -= (10**(x-1))
        for idx_elements in range(open_xs_white.shape[1]):
            start_pos_1, end_pos_1, start_pos_2, end_pos_2 = open_xs_white[idx_x][idx_elements]
            if start_pos_1 == -1:
                break
            else:
                score -= 2*(10**(x-1))
    
    number_forks_black, number_forks_white = forks(open_xs_black[1], open_xs_white[1])
    if number_forks_black > 1:
        score += 2_000 * number_forks_black
    if number_forks_white > 1:
        score -= 2_000 * number_forks_white - 2_000
    return score
