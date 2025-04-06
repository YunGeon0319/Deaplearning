import argparse
import wandb


import numpy as np # 행렬연산
import matplotlib.pyplot as plt # 시각화

import torch # 파이토치
import torch.nn as nn # 파이토치 모듈
import torch.nn.init as init # 초기화 관련 모듈 
import torch.optim as optim #최적화함수
from torch.utils.data import DataLoader # 데이터셋을 학습에 용이하게 바꿈
from torch.utils.data import Dataset
import torch.nn.functional as F # 자주 쓰는 함수를 F로 따로 가져옴
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# In[6]:


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--csv_path', type=str, default='log.csv')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--node_num', type=int, default=100)
args = parser.parse_args()

seed = args.seed
batch_size = args.batch_size
lr = args.lr
epochs = args.epochs
dropout = args.dropout
node_num = args.node_num

wandb.init(
    project="wetie_dl",
    name=f"lr={args.lr}_dropout={args.dropout}_seed={args.seed}",
    config=vars(args))
config = wandb.config

# In[7]:


df = pd.read_csv("C:/Users/user/wetie/3.31/log.csv")  # 너의 CSV 파일 경로
X = df.drop("score", axis=1).values
y = df["score"].values

scaler = StandardScaler()
X = scaler.fit_transform(X) ## 정규화 시키는 과정

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).view(-1, 1) ## tensor으로의 변환



## Dataset 클래스 만들기
class DiabetesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx] 


## 데이터셋 분리 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=config.seed) 
# 전체의 20%만을 테스트셋으로 사용한다. 

train_dataset = DiabetesDataset(X_train, y_train)
test_dataset = DiabetesDataset(X_test, y_test)

train_loader = DataLoader(train_dataset,
                          batch_size=config.batch_size,
                          shuffle=True)
test_loader = DataLoader(test_dataset,
                         batch_size=config.batch_size,
                         shuffle=False)

##MLP 모델 정의


class MLP(nn.Module): # 메타클래스 nn.Module 상속받아 MLP 클래스 생성 (몰라도됨)
	def __init__(self): 
		super(MLP, self).__init__()

class MLP(nn.Module):
    def __init__(self, node_num, dropout):
        super(MLP, self).__init__()
        # layer 생성
        self.fc1 = nn.Linear(5, node_num)
        # input 값은 ABCDE  이렇게 5개이다! => 하이퍼파라미터를 설정할 필요가 없음
        self.fc2 = nn.Linear(node_num, 200) ## node_num은 우리가 바꿔줄 수 있는 거구나!!!
        self.fc3 = nn.Linear(200, 150)
        self.fc4 = nn.Linear(150, 30)
        self.fc5 = nn.Linear(30, 1)  #회귀선은 결국 한 개만 필요하니까!!)
        
        self.dropout = nn.Dropout(dropout) # 연산마다 50% 비율로 랜덤하게 노드 삭제... 나중에
        
        self.batch_norm1 = nn.BatchNorm1d(node_num) # 1dimension이기 때문에 BatchNorm1d를 사용함.
        self.batch_norm2 = nn.BatchNorm1d(200)
        self.batch_norm3 = nn.BatchNorm1d(150)
        self.batch_norm4 = nn.BatchNorm1d(30)

    def forward(self, x): # 모델의 연산 순서를 정의
    # 1st layer
        x = x.view(-1, 5)
        x = self.fc1(x)
        x = self.batch_norm1(x)
        x = F.relu(x) # activation function
        x = self.dropout(x)
    # 2nd layer
        x = self.fc2(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.dropout(x)
    # 3nd layer
        x = self.fc3(x)
        x = self.batch_norm3(x)
        x = F.relu(x)
        x = self.dropout(x)
    # 4nd layer
        x = self.fc4(x)
        x = self.batch_norm4(x)
        x = F.relu(x)
        x = self.dropout(x)
    # 5rd layer
        x = self.fc5(x)
        return x


# In[5]:

model = MLP(node_num=config.node_num, dropout=config.dropout)
def weight_init(m):
    if isinstance(m, nn.Linear):
        init.kaiming_uniform_(m.weight.data)

model= MLP(node_num=config.node_num, dropout=config.dropout)
model.apply(weight_init)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
loss_fn = nn.MSELoss() ##회귀함수이기 때문에 MSE를 사용함!!!


##train 함수
def train(model, train_loader, optimizer, epoch):
    model.train()
    running_loss = 0.0
    n = 0

    for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(X_batch)
        loss = loss_fn(output, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X_batch.size(0)
        n += X_batch.size(0)

    avg_loss = running_loss / n
    wandb.log({"train_loss": avg_loss, "epoch": epoch})
    return avg_loss

##평가 함수

def evaluate(model, test_loader):
    model.eval() # 모델 상태를 '평가중' 으로 변경
    test_loss = 0
    y_preds = []
    x_trues = []

    with torch.no_grad(): # 모델의 평가 단계이므로 gradient가 업데이트 되지 않도록 함
        for score, predict in test_loader:
            output = model(score)
            test_loss += loss_fn(output, predict).item() # loss 누적
            y_preds.append(output.numpy())
            x_trues.append(predict.numpy())

    # numpy 배열로 이어붙이기
    y_pred = np.vstack(y_preds)
    y_true = np.vstack(x_trues)
    
    test_loss /= len(test_loader.dataset) # loss 평균값 계산
    r2 = r2_score(y_true, y_pred) #r^2결정계수 구하기
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    return test_loss, r2, rmse


# In[8]:


# 결과를 저장할 리스트
epoch_list = []
test_loss_list = []
rmse_list = []
r2_list = []

for epoch in range(1, config.epochs + 1):
    train(model, train_loader, optimizer, epoch)
    test_loss, r2, rmse = evaluate(model, test_loader)

    wandb.log({
        "test_loss": test_loss,
        "r2_score": r2,
        "rmse": rmse,
        "epoch": epoch})

    
    epoch_list.append(epoch)
    test_loss_list.append(test_loss)
    rmse_list.append(rmse)
    r2_list.append(r2)
    
    print("[EPOCH: {}], \tTest Loss: {:.4f}, \tRMSE: {:.6f}, \tR2: {:.5f}%".format(
        epoch, test_loss , rmse , r2
    ))


# In[9]:


# 시각화
plt.figure(figsize=(10, 5))

# Test Loss
plt.subplot(1, 3, 1)
plt.plot(epoch_list, test_loss_list, label='Test Loss', color='red')
plt.xlabel('Epoch')
plt.ylabel('Test Loss')
plt.title('Test Loss per Epoch')

# RMSE
plt.subplot(1, 3, 2)
plt.plot(epoch_list, rmse_list, label='RMSE', color='blue')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.title('RMSE per Epoch')

# R2
plt.subplot(1, 3, 3)
plt.plot(epoch_list, r2_list, label='Test Accuracy', color='green')
plt.xlabel('Epoch')
plt.ylabel('R2')
plt.title('R2 per Epoch')

plt.tight_layout()
plt.show()

