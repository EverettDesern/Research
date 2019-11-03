
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class MotionDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.sample = pd.read_csv(csv_file)
        self.transform = transform
        
    def __len__(self):
        return len(self.sample)

    def __getitem__(self, idx):
        landmarks = self.sample.iloc[idx, 0:]
        landmarks = np.array(landmarks)
        # 2x3 : -12 2, 6, 6 -12
        # 3x5 : -28 2, 15, 15 -28
        input = landmarks[:-27]
        input = np.reshape(input, [2,15,15])
        output = landmarks[-27:]
        return {
            "input": input,
            "output": output,
        }

if __name__ == '__main__':

    train = "train.csv"
    dataset = MotionDataset(train)

    dataset[0]


