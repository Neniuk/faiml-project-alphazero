import chess

def print_board_indices():
    board_indices = [[0 for _ in range(8)] for _ in range(8)]
    for i in range(64):
        row = 7 - (i // 8)
        col = i % 8
        board_indices[row][col] = i

    for row in board_indices:
        print(" ".join(f"{sq:2}" for sq in row))

def print_chess_board(board):
    board_str = str(board)
    board_rows = board_str.split('\n')
    for row in board_rows:
        print(row)

def print_board_with_indices_and_pieces(board):
    board_indices = [[0 for _ in range(8)] for _ in range(8)]
    for i in range(64):
        row = 7 - (i // 8)
        col = i % 8
        board_indices[row][col] = i

    for row in range(8):
        for col in range(8):
            index = board_indices[row][col]
            piece = board.piece_at(index)
            piece_symbol = piece.symbol() if piece else '.'
            print(f"{index:2}:{piece_symbol}", end=" ")
        print()

def get_square_index(square):
    return chess.SQUARE_NAMES.index(square)

# Create a new chess board
board = chess.Board()

print("Board Indices:")
print_board_indices()
print("\nActual Chess Board:")
print_chess_board(board)
print("\nBoard with Indices and Pieces:")
print_board_with_indices_and_pieces(board)

# Test the index of h8
square = "h8"
index = get_square_index(square)
print(f"\nIndex of {square}: {index}")

# Output print_board_indices:
# 56 57 58 59 60 61 62 63
# 48 49 50 51 52 53 54 55
# 40 41 42 43 44 45 46 47
# 32 33 34 35 36 37 38 39
# 24 25 26 27 28 29 30 31
# 16 17 18 19 20 21 22 23
#  8  9 10 11 12 13 14 15
#  0  1  2  3  4  5  6  7

# Actual Chess Board:
# r n b q k b n r
# p p p p p p p p
# . . . . . . . .
# . . . . . . . .
# . . . . . . . .
# . . . . . . . .
# P P P P P P P P
# R N B Q K B N R

# Board with Indices and Pieces:
# 56:r 57:n 58:b 59:q 60:k 61:b 62:n 63:r 
# 48:p 49:p 50:p 51:p 52:p 53:p 54:p 55:p 
# 40:. 41:. 42:. 43:. 44:. 45:. 46:. 47:. 
# 32:. 33:. 34:. 35:. 36:. 37:. 38:. 39:. 
# 24:. 25:. 26:. 27:. 28:. 29:. 30:. 31:. 
# 16:. 17:. 18:. 19:. 20:. 21:. 22:. 23:. 
# 8:P  9:P 10:P 11:P 12:P 13:P 14:P 15:P 
# 0:R  1:N  2:B  3:Q  4:K  5:B  6:N  7:R 
