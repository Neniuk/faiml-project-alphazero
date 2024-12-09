import sys
import os

# Add the chess_game directory to PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), 'chess_game'))

from ChessGame import ChessGame
from pytorch.NNet import ChessNNWrapper
import numpy as np

def main():
    game = ChessGame()
    nnet = ChessNNWrapper(game)

    # Generate some dummy training data
    examples = []
    for _ in range(100):  # Number of examples
        board = game.getInitBoard()
        pi = np.random.rand(game.getActionSize())
        v = np.random.rand()
        examples.append((game.boardToTensor(board), pi, v))

    nnet.train(examples)

    # Test prediction
    board = game.getInitBoard()
    pi, v = nnet.predict(board)
    print("Predicted policy:", pi)
    print("Predicted value:", v)

if __name__ == "__main__":
    main()