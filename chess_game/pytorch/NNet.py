import torch
import torch.nn as nn
import torch.optim as optim
from NeuralNet import NeuralNet

class ChessNNet(nn.Module):
    def __init__(self, board_size, action_size):
        super(ChessNNet, self).__init__()
        self.board_x, self.board_y = board_size
        self.conv1 = nn.Conv2d(12, 64, 3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * self.board_x * self.board_y, 1024)
        self.fc2 = nn.Linear(1024, action_size)

    def forward(self, board):
        x = torch.relu(self.conv1(board))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        policy = self.fc2(x)
        return policy, torch.tanh(self.fc2(x))  # Policy and Value

class ChessNNWrapper(NeuralNet):
    def __init__(self, game):
        self.nnet = ChessNNet(game.getBoardSize(), game.getActionSize())
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

        if torch.cuda.is_available():
            self.nnet.cuda()

    def train(self, examples):
        optimizer = optim.Adam(self.nnet.parameters())
        for epoch in range(10):  # Number of epochs
            print('EPOCH ::: ' + str(epoch + 1))
            self.nnet.train()
            for batch_idx, (boards, pis, vs) in enumerate(examples):
                boards = torch.tensor(boards, dtype=torch.float32)
                pis = torch.tensor(pis, dtype=torch.float32)
                vs = torch.tensor(vs, dtype=torch.float32)

                if torch.cuda.is_available():
                    boards, pis, vs = boards.cuda(), pis.cuda(), vs.cuda()

                optimizer.zero_grad()
                out_pi, out_v = self.nnet(boards)
                l_pi = nn.functional.cross_entropy(out_pi, pis)
                l_v = nn.functional.mse_loss(out_v, vs)
                total_loss = l_pi + l_v
                total_loss.backward()
                optimizer.step()

    def predict(self, board):
        board_tensor = torch.tensor(self.game.boardToTensor(board), dtype=torch.float32).unsqueeze(0)
        if torch.cuda.is_available():
            board_tensor = board_tensor.cuda()
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board_tensor)
        return torch.softmax(pi, dim=1).cpu().numpy()[0], v.cpu().numpy()[0]