import torch #torchはPyTorchのモジュール
import numpy as np
import torch.nn.functional as F #torch.nn.functionalはPyTorchのニューラルネットワークの機能を提供するモジュール
import torchvision
from torch.utils.tensorboard import SummaryWriter #SummaryWriterはTensorBoardのログを書き込むためのモジュール
from tqdm.auto import tqdm #tqdmはプログレスバーを表示するためのモジュール

torch.manual_seed(0) #乱数のシードを固定
torch.cuda.manual_seed(0) #乱数のシードを固定
torch.backends.cudnn.enabled = True #CuDNNを使用する
torch.backends.cudnn.benchmark = True #CuDNNのベンチマークを有効にする

#計算デバイスはGPUを使用する
device = torch.device('cuda:0')

#データの前処理
pre_process = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(), #テンソルに変換
        torchvision.transforms.Normalize((0.5,), (0.5,)) ,#正規化
        torchvision.transforms.Lambda(lambda x: x.view(-1)) #1次元に変換
    ]
)

#MNISTのデータセットをダウンロード
train_dataset = torchvision.datasets.MNIST(
    root = "./data",
    train = True,
    download = True,    
    transform = pre_process
)

test_dataset = torchvision.datasets.MNIST(
    root = "./data",
    train = False,
    download = True,
    transform = pre_process
)

#データローダーの作成
train_loader = torch.utils.data.DataLoader(
    dataset = train_dataset,
    batch_size = 512,
    shuffle = True,
    num_workers = 2,
    drop_last = True,
)

test_loader = torch.utils.data.DataLoader(
    dataset = test_dataset,
    batch_size = 512,
    shuffle = False,
    num_workers = 2,
    drop_last = False
)

#モデルの定義
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 128)
        self.fc2 = torch.nn.Linear(128, 10)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x) #活性化関数ReLU
        return self.fc2(x)
    
#モデルのインスタンス化
model = Net().to(device)

#損失関数の定義
criterion = torch.nn.CrossEntropyLoss()

#オプティマイザの定義
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

#繰り返し回数
epochs = 10

#評価結果の保存用
results = np.zeros((0, 5))

for epoch in tqdm(range(epochs), desc="Epochs"):
    n_train_acc, train_loss = 0.0, 0.0
    n_test_acc, test_loss = 0.0, 0.0
    num_train, num_test = 0, 0

    for images, labels in tqdm(
        train_loader,
        total = len(train_loader),
        leave=False,
        desc = "Training" ,
    ):
        
        # 1バッチあたりのデータ件数
        train_batch_size = len(labels)
        # 1エポックあたりのデータ累積件数
        num_train += train_batch_size


        images, labels = images.to(device), labels.to(device)

        #勾配を初期化
        optimizer.zero_grad()
        #順伝播
        outputs = model(images)
        #損失関数の計算
        loss = criterion(outputs, labels)
        #逆伝播
        loss.backward()
        #パラメータの更新
        optimizer.step()

        predicted = torch.max(outputs.data, 1)[1]

        train_loss += loss.item() * labels.size(0) #labels.size(0)はバッチサイズ, .size(1)は次元数
        n_train_acc += (predicted == labels).sum().item() 

    
    for test_images, test_labels in tqdm(
        test_loader,
        total = len(test_loader),
        leave = False,
        desc = "Test",
    ):
        test_images, test_labels = test_images.to(device), test_labels.to(device)

        #順伝播
        outputs = model(test_images)
        #損失関数の計算
        loss = criterion(outputs, test_labels)
        #予測
        predicted = torch.max(outputs.data, 1)[1]
        #正解数のカウント
        num_test += test_labels.size(0)
        n_test_acc += (predicted == test_labels).sum().item()
        #損失の累積
        test_loss += loss.item() * test_labels.size(0)

    # 精度計算
    train_acc = n_train_acc / num_train
    val_acc = n_test_acc / num_test
    #損失計算
    ave_train_loss = train_loss / num_train
    ave_val_loss = test_loss / num_test
    # 結果表示
    #print (f'Epoch [{epoch+1}/{epochs}], loss: {ave_train_loss:.5f} acc: {train_acc:.5f} val_loss: {ave_val_loss:.5f}, val_acc: {val_acc:.5f}')
    # 記録
    item = np.array([epoch+1 , ave_train_loss, train_acc, ave_val_loss, val_acc])
    results = np.vstack((results, item))


