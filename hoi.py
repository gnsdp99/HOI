import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from pytorchtools import EarlyStopping

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# dataset
df = pd.read_table('new_label.txt', sep=' ', header=None, names=['class', 'top_obj', 'mid_obj', 'bot_obj'])

X = df.drop('class', axis=1).to_numpy()
Y = df['class'].to_numpy().reshape((-1, 1))

# # data normalization
# scaler = MinMaxScaler()
# scaler.fit(X)
# X = scaler.transform(X)
# scaler.fit(Y)
# Y = scaler.transform(Y)

# # data to tensor
# class TensorData(Dataset):
#     def __init__(self, x_data, y_data):
#         self.x_data = torch.FloatTensor(x_data)
#         self.y_data = torch.FloatTensor(y_data)
#         self.len = self.y_data.shape[0]
    
#     def __getitem__(self, index):
#         return self.x_data[index], self.y_data[index]

#     def __len__(self):
#         return self.len

# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5)

# trainsets = TensorData(X_train, y_train)
# trainloader = torch.utils.data.DataLoader(trainsets, batch_size=32, shuffle=True)

# testsets = TensorData(X_test, y_test)
# testloader = torch.utils.data.DataLoader(testsets, batch_size=32, shuffle=False)

# # Neural Network
# class NeuralNetwork(nn.Module):
#     def __init__(self):
#         super(NeuralNetwork, self).__init__() # nn.Module 부모 클래스 상속
#         self.layer = nn.Sequential( # layer를 쌓을 때 사용
#             nn.Linear(4, 2, bias=True), # 입력층 -> 은닉층1
#             nn.ReLU(),
#             nn.Linear(2, 2, bias=True), # 은닉층1 -> 은닉층2
#             nn.ReLU(),
#             nn.Linear(2, 1, bias=True) # 은니층2 -> 출력층
#         )
        
#     def forward(self, x): # 모델에 input을 넣으면 따로 호출하지 않아도 자동으로 호출된다. ...(1)
#         out = self.layer(x)
#         return out

# # Train Function
# def train(dataloader, model, loss_fn, optimizer):
#     pbar = tqdm(dataloader, desc=f'Training') # 진행률 bar
#     for batch, (X, y) in enumerate(pbar): # batch 단위로 training
#         X, y = X.to(device), y.to(device)
#         pred = model(X) # predict ...(1)
#         loss = loss_fn(pred, y) # calculate loss
        
#         # backpropagation
#         optimizer.zero_grad() # backward 할 때마다 gradient를 더해주기 때문에 batch마다 초기화해야 한다.
#         loss.backward() # gradient 계산 함수. loss에 대해 모든 가중치를 미분하므로 시작지점인 loss에 적용해야 한다.
#         optimizer.step() # network의 parameter(weight, bias)를 업데이트 하는 함수
        
# # Test Function
# def test(dataloader, model, loss_fn):
#     size = len(dataloader.dataset)
#     num_batches = len(dataloader)
#     model.eval() # evaluation 과정에서 사용하지 않아야 하는 layer들을 알아서 off 시키는 함수. evaluation이 끝나면 다시 train mode로 변경해야 한다. training때 필요하지만 inference에는 필요없는 layer가 있다.
#     loss, correct = 0, 0
#     with torch.no_grad(): # inference시에는 autograd를 끔으로써 메모리 사용량을 줄이고 연산 속도를 높인다.
#         for X, y in dataloader:
#             X, y = X.to(device), y.to(device)
#             pred = model(X) # pred의 output은 batch_size x num_labels의 size를 갖는 tensor이다.
#             loss += loss_fn(pred, y).item() # item(): tensor의 scalar값을 구한다.
#             correct += (pred.argmax(1) == y).type(torch.float).sum().item()
#     loss /= num_batches
#     correct /= size
#     print(f'Test Accuracy: {(100*correct):>0.1f}%   Loss: {loss:>8f} \n')
    
#     return 100*correct, loss

# # Generate the NN Model
# model = NeuralNetwork().to(device)

# # Set the Training Parameters
# lr = 1e-3
# loss_fn = nn.CrossEntropyLoss().to(device)
# optimizer = optim.SGD(model.parameters(), lr=lr) # gradient가 0에 가까워지도록 하는 optimization 함수. parameters 함수는 NeuralNetwork의 weight/bias를 리턴한다.
# early_stopping = EarlyStopping(patience=5, verbose=True) # 5번안에 감소되지 않으면 stop (overfitting 방지)

# # Plot
# fig = plt.figure(figsize=(20, 5))
# line1, line2 = plt.plot([], [], [], [])
# plt.clf()

# # Train the Network
# epochs = 100
# for t in range(epochs):
#     print(f'----- Epoch {t+1} -----')
#     train(trainloader, model, loss_fn, optimizer)
#     val_acc, val_loss = test(testloader, model, loss_fn)
    
#     # Add Accuracy, Loss to the lines
#     line1.set_xdata(np.append(line1.get_xdata(), t+1))
#     line1.set_ydata(np.append(line1.get_ydata(), val_loss))
#     line2.set_ydata(np.append(line2.get_ydata(), val_acc))
    
#     early_stopping(val_loss, model) # overfitting 상황 tracking
#     if early_stopping.early_stop:
#         break
    
# fig.add_subplot(1,2,1)
# plt.plot(line1.get_xdata(), line1.get_ydata(), color='red')
# plt.plot(line1.get_xdata(), line1.get_ydata(), 'o', color='red')
# plt.xlabel('Epoch', fontsize=12); plt.ylabel('Validation Loss', fontsize=12)
# fig.add_subplot(1,2,2)
# plt.plot(line1.get_xdata(), line2.get_ydata(), color='blue')
# plt.plot(line1.get_xdata(), line2.get_ydata(), 'o', color='blue')
# plt.xlabel('Epoch', fontsize=12); plt.ylabel('Validation Accuracy', fontsize=12)
 
# plt.tight_layout()
# plt.autoscale()
# plt.show()

# torch.save(model, 'hoi.pt') # save model