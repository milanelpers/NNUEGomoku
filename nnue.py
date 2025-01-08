import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

class GomokuDataset(Dataset):
    def __init__(self, player1_data, player2_board, labels):
        self.player1_data = player1_data
        self.player2_board = player2_board
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.player1_data[idx], self.player2_board[idx], self.labels[idx]
    

class NNUEGomoku(torch.nn.Module):
    def __init__(self):
        super(NNUEGomoku, self).__init__()
        self.input_layer_p1 = torch.nn.Linear(225, 512)
        self.input_layer_p2 = torch.nn.Linear(225, 512)
        self.hidden_layer = torch.nn.Linear(1024, 64)
        self.output_layer = torch.nn.Linear(64, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, player1_board, player2_board):
        x1 = self.input_layer_p1(player1_board)
        x2 = self.input_layer_p2(player2_board)
        x1 = self.relu(x1)
        x2 = self.relu(x2)
        x = torch.cat((x1, x2), dim=1)
        x = self.hidden_layer(x)
        x = self.relu(x)
        x = self.output_layer(x)
        return x


def train(model, train_loader, test_loader, criterion, optimizer, num_epochs=30):
    train_losses = []
    val_losses = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_loss = 0.0
        for player1_board, player2_board, labels in train_loader:
            player1_board, player2_board, labels = player1_board.to(device), player2_board.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(player1_board, player2_board)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for player1_board, player2_board, labels in test_loader:
                outputs = model(player1_board, player2_board)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_loss /= len(test_loader)
        val_losses.append(val_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    torch.save(model.state_dict(), 'nnue_gomoku_model.pth')
    return train_losses, val_losses


def main():
    batch_size = 64
    lr = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    player1_boards = torch.load('player1_boards.pt', weights_only=True).to(torch.float32).to(device)
    player2_boards = torch.load('player2_boards.pt', weights_only=True).to(torch.float32).to(device)
    labels = torch.load('labels.pt', weights_only=True).to(torch.float32).to(device)

    player1_boards_train, player1_boards_test, player2_boards_train, player2_boards_test, labels_train, labels_test = train_test_split(player1_boards, player2_boards, labels, test_size=0.2, shuffle=True)

    train_dataset = GomokuDataset(player1_boards_train, player2_boards_train, labels_train)
    test_dataset = GomokuDataset(player1_boards_test, player2_boards_test, labels_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = NNUEGomoku()
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.to(device)

    start = time.perf_counter()
    train_losses, val_losses = train(model, train_loader, test_loader, criterion, optimizer)
    print(f'Training time: {time.perf_counter() - start:.2f} seconds')

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss Over Time')
    plt.savefig('training_validation_loss.png')


if __name__ == '__main__':
    main()