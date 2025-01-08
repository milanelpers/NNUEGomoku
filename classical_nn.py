import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm


class ClassicalNN(nn.Module):
    def __init__(self):
        super(ClassicalNN, self).__init__()

        self.fc1 = nn.Linear(225, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class GomokuDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]



def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path, dtype={'board': 'str', 'label': 'int'}, engine='pyarrow')
    df.drop_duplicates(inplace=True)
    df = df[df["number_moves"] <= 225]
    df = df[(df["label"] >= -5_000) & (df["label"] <= 5_000)]
    df = df.drop(columns=['number_moves'])
    df["board"] = df["board"].apply(lambda x: np.fromstring(x, dtype=np.int8, sep=" "))

    df = df.sample(frac=1).reset_index(drop=True)

    boards = torch.from_numpy(np.vstack(df["board"])).to(torch.float32)
    labels = torch.from_numpy(np.vstack(df["label"].values.astype(np.int16))).to(torch.float32)
    return boards, labels



def train(model, train_loader, val_loader, criterion, optimizer, epochs=30):
    train_losses = []
    val_losses = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for epoch in tqdm(range(epochs)):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    return train_losses, val_losses


def main():
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 30

    file_path = 'train.csv'
    data, labels = load_and_preprocess_data(file_path)


    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, shuffle=True)

    train_dataset = GomokuDataset(train_data, train_labels)
    test_dataset = GomokuDataset(test_data, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    model = ClassicalNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_losses, val_losses = train(model, train_loader, test_loader, criterion, optimizer, epochs=num_epochs)

    torch.save(model.state_dict(), 'classical_nn_model.pth')

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss Over Time')
    plt.savefig('training_validation_loss_classical_nn.png')

if __name__ == "__main__":
    main()