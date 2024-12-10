from Game import Game
import chess
import numpy as np
import logging

class ChessGame(Game):
    def getInitBoard(self):
        board = chess.Board()
        return board

    def getBoardSize(self):
        return (8, 8)  # Chess board size

    def getActionSize(self):
        return 4672  # Number of legal moves in chess
    
    def getSymmetries(self, board, pi):
        """
        Get all symmetrical forms of the board and policy vector.
        """
        assert len(pi) == self.getActionSize()  # Ensure the policy vector has the correct size
        symmetries = []

        # Add the original board and policy
        symmetries.append((board, pi))

        # Add the mirrored board and policy
        mirrored_board = board.mirror()
        mirrored_pi = pi[::-1]  # Reverse the policy vector for the mirrored board
        symmetries.append((mirrored_board, mirrored_pi))

        return symmetries

    def getNextState(self, board, player, action):
        if board is None:
            raise ValueError("Board is None in getNextState")
        move = self.index_to_move(action)
        logging.info(f"Generated move: {move.uci()} for action: {action}")
        logging.info(f"Board before move:\n{board}")
        board.push(move)
        logging.info(f"Board after move:\n{board}")
        return board, -player

    def index_to_move(self, index):
        from_square = index // 64
        to_square = index % 64
        promotion_index = (index // 4096) % 5
        promotion = {0: '', 1: 'q', 2: 'r', 3: 'b', 4: 'n'}[promotion_index]

        # Ensure from_square and to_square are within valid range
        if from_square < 0 or from_square >= 64 or to_square < 0 or to_square >= 64:
            raise ValueError(f"Invalid square index: from_square={from_square}, to_square={to_square}")

        move_uci = chess.SQUARE_NAMES[from_square] + chess.SQUARE_NAMES[to_square] + promotion

        # Handle castling moves
        if from_square == chess.E1 and to_square == chess.G1:
            move_uci = "e1g1"  # Kingside castling for white
        elif from_square == chess.E1 and to_square == chess.C1:
            move_uci = "e1c1"  # Queenside castling for white
        elif from_square == chess.E8 and to_square == chess.G8:
            move_uci = "e8g8"  # Kingside castling for black
        elif from_square == chess.E8 and to_square == chess.C8:
            move_uci = "e8c8"  # Queenside castling for black

        # Handle en passant moves
        if promotion == '' and abs(from_square - to_square) in [7, 9] and (chess.square_rank(to_square) in [2, 5]):
            move_uci = chess.SQUARE_NAMES[from_square] + chess.SQUARE_NAMES[to_square]

        logging.info(f"Generated move UCI: {move_uci} for index: {index}")
        move = chess.Move.from_uci(move_uci)
        logging.info(f"Move object: {move}")
        return move

    def getCanonicalForm(self, board, player):
        if player == 1:
            return board
        return board.mirror()

    def getValidMoves(self, board, player):
        valid_moves = [0] * self.getActionSize()
        for move in board.legal_moves:
            idx = self.move_to_index(board, move)
            valid_moves[idx] = 1
        return valid_moves
    
    def move_to_index(self, board, move):
        # Convert the move to an index
        move_uci = move.uci()
        from_square = chess.SQUARE_NAMES.index(move_uci[:2])
        to_square = chess.SQUARE_NAMES.index(move_uci[2:4])
        promotion = move_uci[4:] if len(move_uci) > 4 else ''
        promotion_index = {'': 0, 'q': 1, 'r': 2, 'b': 3, 'n': 4}.get(promotion, 0)

        # Handle castling moves
        if move in [chess.Move.from_uci("e1g1"), chess.Move.from_uci("e1c1"), chess.Move.from_uci("e8g8"), chess.Move.from_uci("e8c8")]:
            if move == chess.Move.from_uci("e1g1"):
                from_square = chess.E1
                to_square = chess.G1
            elif move == chess.Move.from_uci("e1c1"):
                from_square = chess.E1
                to_square = chess.C1
            elif move == chess.Move.from_uci("e8g8"):
                from_square = chess.E8
                to_square = chess.G8
            elif move == chess.Move.from_uci("e8c8"):
                from_square = chess.E8
                to_square = chess.C8

        # Handle en passant moves
        if board.is_en_passant(move):
            to_square = move.to_square

        return from_square * 64 + to_square + promotion_index * 4096

    def getGameEnded(self, board, player):
        if board.is_checkmate():
            return 1 if player == 1 else -1
        if board.is_stalemate() or board.is_insufficient_material():
            # draw has a very little value 
            return 1e-4
        return 0

    def stringRepresentation(self, board):
        if board is not None:
            return board.fen()  # Use FEN for serialization
        else:
            raise ValueError("Board is None")

    def boardToTensor(self, board):
        piece_map = board.piece_map()
        tensor = np.zeros((12, 8, 8), dtype=np.float32)
        for square, piece in piece_map.items():
            piece_type = piece.piece_type - 1
            color = 0 if piece.color else 6
            tensor[color + piece_type, square // 8, square % 8] = 1
        return tensor
