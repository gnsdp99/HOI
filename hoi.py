import os
import torch
from torch import nn
from torch import optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# dataset
hoi_dict = [
    {'label': 0, 'x1': 10, 'x2': 20, 'x3': 40},
    {'label': 1, 'x1': 30, 'x2': 10, 'x3': 20},
    {'label': 2, 'x1': 40, 'x2': 20, 'x3': 10}
]
df = pd.DataFrame(hoi_dict)

X = df.drop('label', axis=1).to_numpy()
Y = df['label'].to_numpy().reshape((-1, 1))

# data normalization
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)
scaler.fit(Y)
Y = scaler.transform(Y)

# data to tensor
class TensorData(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = torch.FloatTensor(x_data)
        self.y_data = torch.FloatTensor(y_data)
        self.len = self.y_data.shape[0]
    
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5)


trainsets = TensorData(X_train, y_train)
trainloader = torch.utils.data.DataLoader(trainsets, batch_size=32, shuffle=True)

testsets = TensorData(X_test, y_test)
testloader = torch.utils.data.DataLoader(testsets, batch_size=32, shuffle=False)

class Classifier(nn.Module):
    def __init__(self):
        self.fc1 = nn.Linear(13, 50, bias=True) # 입력층 -> 은닉층1
        self.fc2 = nn.Linear(50, 30, bias=True) # 은닉층1 -> 은닉층2
        self.fc3 = nn.Linear(30, 1, bias=True) # 은니층2 -> 출력층
        self.dropout = nn.Dropout(0, 2) # 20% 비율로 랜덤하게 노드 제거
        
    def forward(self, x): # 연산 순서
        x = F.relu(self.fc1(x)) # 입력층 -> ReLU
        x = self.dropout(F.relu(self.fc2(x))) # 은닉층 
        x = F.relu(self.fc3(x))
        return x