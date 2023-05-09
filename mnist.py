import torch #torchはPyTorchのモジュール
import numpy as np
import torch.nn.functional as F #torch.nn.functionalはPyTorchのニューラルネットワークの機能を提供するモジュール
import torchvision
from torch.utils.tensorboard import SummaryWriter #SummaryWriterはTensorBoardのログを書き込むためのモジュール
from tqdm.auto import tqdm #tqdmはプログレスバーを表示するためのモジュール
import matplotlib.pyplot as plt #matplotlib.pyplotはグラフを描画するためのモジュール


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

#データローダーの作成, データローダーはデータセットからミニバッチを作成する
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

# imagesは４次元テンソルであり、(batch_size, channels, height, width)の形状、labelsは1次元テンソルであり、(batch_size,)の形状をしている。
    for images, labels in tqdm(
        train_loader,
        total = len(train_loader),
        leave=False, #ターミナル上にプログレスバーを残すかどうか
        desc = "Training" ,
    ):
        
        # 1バッチあたりのデータ件数
        train_batch_size = len(labels)
        # print(train_batch_size) #debug == 512

        # 1エポックあたりのデータ累積件数
        num_train += train_batch_size #バッチ処理を行うたびに、num_trainにtrain_batch_sizeを加算し、訓練データの累積件数を計算している


        images, labels = images.to(device), labels.to(device)

        """
        print(images.shape) #debug == torch.Size([512, 784]) == 512枚の画像データが784次元のベクトルに変換されている
        print(labels.shape) #debug == torch.Size([512]) == 512枚の画像データに対応するラベルデータ
        """

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

        """
        torch.max()は、指定したテンソルの最大値とその位置を返す関数。第1引数には対象となるテンソルを指定し、第2引数には最大値を求める軸を指定する。
        outputs.dataは、モデルの出力として得られたテンソルであり、それぞれの行に対して、各クラスに属する確率が格納されています。
        このテンソルの最大値を求める軸を1に指定することで、「各行（サンプル）ごとに最も確率の高いクラスの値とその位置（インデックス）」を求めることができます。
        [1]は、torch.max()の返り値から位置（インデックス）を取り出すために使用されます。torch.max()は最大値とその位置の2つの値を返すので、位置を取り出すためには1を指定します。
        したがって、torch.max(outputs.data, 1)[1]は、各サンプルに対する最も確率の高いクラスのインデックスを表します。
        """

        predicted = torch.max(outputs.data, 1)[1] 
        #print(predicted) #debug

        train_loss += loss.item() * labels.size(0) #loss.item()は現在のバッチのロスの平均値だからバッチサイズをかけてる。labels.size(0)はバッチサイズを返す
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
        num_test += test_labels.size(0) #テストバッチサイズが終わったら、そのサイズ分をテスト数に足す
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

# 結果のグラフ化
# 
# 1行2列のグラフを作成
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
# 1列目にはlossの推移
ax[0].plot(results[:, 0], results[:, 1], label='train loss')
ax[0].plot(results[:, 0], results[:, 3], label='test loss')
ax[0].set_xlabel('epoch')
ax[0].set_ylabel('loss')
ax[0].legend()
# 2列目にはaccuracyの推移
ax[1].plot(results[:, 0], results[:, 2], label='train acc')
ax[1].plot(results[:, 0], results[:, 4], label='test acc')
ax[1].set_xlabel('epoch')
ax[1].set_ylabel('acc')
ax[1].legend()
# グラフを表示
plt.savefig('result.png')



