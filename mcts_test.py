import chess
import numpy as np
from MCTS import MCTS
from chess_game.ChessGame import ChessGame
from chess_game.pytorch.NNet import NNetWrapper as NNet
from utils import dotdict

# Initialize the game and neural network
game = ChessGame()
nnet = NNet(game)

# Load a pretrained model if available
# nnet.load_checkpoint('./pretrained_models/chess/pytorch/', 'best.pth.tar')

# Define MCTS arguments
args = dotdict({
    'numMCTSSims': 50,
    'cpuct': 1.0
})

# Initialize MCTS
mcts = MCTS(game, nnet, args)

# Get the initial board state
board = game.getInitBoard()

# Print the initial board state
print("Initial board state:")
print(board)

# Convert the board to the canonical form
canonical_board = game.getCanonicalForm(board, 1)
canonical_board_copy = canonical_board.copy()

# Get the action probabilities from MCTS using a copy of the canonical board
action_probs = mcts.getActionProb(canonical_board_copy, temp=1)

# Select the action with the highest probability
action = np.argmax(action_probs)

print(f"Selected action: {action}")

# Convert the action index to a move
move = game.index_to_move(action)

print("Board before the move:")
print(board)

# Print the move
print(f"Selected move: {move.uci()}")

# Apply the move to the board
board.push(move)

# Print the board after the move
print("Board after the move:")
print(board)
