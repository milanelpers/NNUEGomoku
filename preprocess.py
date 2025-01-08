import numpy as np
import pandas as pd
import torch
import time

start = time.perf_counter()
df = pd.read_csv('train.csv', dtype={'board': 'str', 'number_moves': 'int', 'label': 'int'}, engine='pyarrow')
df = df.iloc[1_000_000:]
df.drop_duplicates(inplace=True)
df = df[df["number_moves"] <= 225]
df = df[(df["label"] >= -5_000) & (df["label"] <= 5_000)]

df["board"] = df["board"].apply(lambda x: np.fromstring(x, dtype=np.int8, sep=" "))


df = df.sample(frac=1).reset_index(drop=True)

boards = torch.from_numpy(np.vstack(df["board"]))
labels = torch.from_numpy(np.vstack(df["label"].values.astype(np.int16)))

player1_boards = (boards == 1)
player2_boards = (boards == 2)

torch.save(player1_boards, 'player1_boards.pt')
torch.save(player2_boards, 'player2_boards.pt')
torch.save(labels, 'labels.pt')

print(f"Length of data: {len(df)}")
print(f"Time for preprocessing: {time.perf_counter() - start} seconds")