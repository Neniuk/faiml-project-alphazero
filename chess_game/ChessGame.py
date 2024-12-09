from Game import Game
import chess
import numpy as np

class ChessGame(Game):
    def getInitBoard(self):
        board = chess.Board()
        return board

    def getBoardSize(self):
        return (8, 8)  # Chess board size

    def getActionSize(self):
        return 4672  # Number of legal moves in chess

    def getNextState(self, board, player, action):
        board.push(action)
        return board, -player

    def getValidMoves(self, board, player):
        valid_moves = [0] * self.getActionSize()
        for move in board.legal_moves:
            valid_moves[move] = 1
        return valid_moves

    def getGameEnded(self, board, player):
        if board.is_checkmate():
            return 1 if player == 1 else -1
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        return None

    def stringRepresentation(self, board):
        return board.fen()  # Use FEN for serialization

    def boardToTensor(self, board):
        piece_map = board.piece_map()
        tensor = np.zeros((12, 8, 8), dtype=np.float32)
        for square, piece in piece_map.items():
            piece_type = piece.piece_type - 1
            color = 0 if piece.color else 6
            tensor[color + piece_type, square // 8, square % 8] = 1
        return tensor