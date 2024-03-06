import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import trange
import os
import variable as var

from real_dataset import create_train_set
from real_dataset import create_test_set

from algorithm_utils import Algorithm, PyTorchUtils


class LSTM_enc_dec(Algorithm, PyTorchUtils):
    def __init__(self, dataset_name: str = {}, name: str = 'LSTM-ED', num_epochs: int = 10, batch_size: int = 20, lr: float = 1e-3,
                 hidden_size: int = 5, sequence_length: int = 30, train_gaussian_percentage: float = 0.25,
                 n_layers: tuple = (1, 1), use_bias: tuple = (True, True), dropout: tuple = (0, 0),
                 seed: int = None, gpu: int = None, details=True):
        Algorithm.__init__(self, __name__, name, seed, details=details)
        PyTorchUtils.__init__(self, seed, gpu)
        self.dataset_name = dataset_name
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr

        self.hidden_size = hidden_size  # 瓶颈层的大小，隐藏层状态的维数，即隐藏层节点的个数
        self.sequence_length = sequence_length
        self.train_gaussian_percentage = train_gaussian_percentage

        self.n_layers = n_layers  # LSTM 堆叠的层数，默认值是1层，如果设置为2，第二个LSTM接收第一个LSTM的计算结果
        self.use_bias = use_bias  # 隐层状态是否带bias，默认为true。bias是偏置值，或者偏移值。没有偏置值就是以0为中轴，或以0为起点。
        self.dropout = dropout  # 默认值0。是否在除最后一个 RNN 层外的其他 RNN 层后面加 dropout 层。输入值是 0-1 之间的小数，表示概率。0表示0概率dropout，即不dropout

        
        self.mean, self.cov = None, None

    def fit(self, model, X: pd.DataFrame):
        train_loader, train_gaussian_loader = create_train_set(X)

        model.to(var.device)
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=self.lr)  # alpha：初始步长（学习率），典型值为0.001。beta1：第一个动量的衰减因子，典型值为0.9。beta2：无穷大范数的衰减因子，典型值为0.999。

        model.train()
        train_loss = []
        val_losses = []
        train_batch_loss = []
        val_batch_loss = []
        for epoch in trange(self.num_epochs):
            for ts_batch in train_loader:
                output = model(ts_batch.to(var.device))
                loss = nn.HuberLoss(reduction='none')(output, ts_batch.float().to(var.device))
                losses = loss.mean()
                model.zero_grad()
                losses.backward()
                optimizer.step()
                train_loss.append(losses.item())
                train_batch_loss.append(losses.item())

            for ts_batch in train_gaussian_loader:
                output = model(ts_batch.to(var.device))
                val_loss = nn.L1Loss(reduce=False)(output, ts_batch.float().to(var.device)).mean()  # [20,5,51]
                val_losses.append(val_loss.item())
                val_batch_loss.append(val_loss.item())

            # print progress
            print(
                "Epoch: %d, Loss %.8f, Validation Loss %.8f" % (epoch, np.mean(train_batch_loss), np.mean(val_batch_loss)))
                
            # early stopping
            if val_loss < 0.003:
                break

        model.eval()

        print(model)
        model_save_file = "/data/liuyr/code/Finally/save_results/models/%s/" % self.dataset_name
        if not os.path.exists(os.path.dirname(model_save_file)):
            os.makedirs(os.path.dirname(model_save_file))

        torch.save(
            {'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'loss': loss,
            'val_loss': val_loss
            }, "%sepoch=%s_net.pth" % (model_save_file,self.num_epochs))
            
        return model


    def predict(self, model,X: pd.DataFrame):
        data_loader = create_test_set(X)

        model.eval()
        outputs = []
        for idx, ts in enumerate(data_loader):
            _, output = model(ts.to(var.device), True)
            outputs.append(output.data.cpu().numpy())


        outputs = np.concatenate(outputs)
        lattice = np.full((self.sequence_length, X.shape[0], X.shape[1]), np.nan)
        for i, output in enumerate(outputs):
            lattice[i % self.sequence_length, i:i + self.sequence_length, :] = output
        outputs = np.nanmean(lattice, axis=0)
        outputs = torch.tensor(outputs)
        return outputs


class LSTMEDModule(nn.Module, PyTorchUtils):
    def __init__(self, n_features: int, hidden_size: int, n_layers: tuple, use_bias: tuple,
                 dropout: tuple, seed: int, gpu: int):
        super().__init__()
        PyTorchUtils.__init__(self, seed, gpu)
        self.n_features = n_features
        self.hidden_size = hidden_size

        self.n_layers = n_layers  # 编码层是一个LSTM，解码层是一个LSTM，LSTM 堆叠的层数为1
        self.use_bias = use_bias  # 隐层状态带偏置值，或者偏移值
        self.dropout = dropout  # 不 在除最后一个 RNN 层外的其他 RNN 层后面加 dropout 层
        # batch_first: 输入输出的第一维是否为 batch_size，默认值 False。batch_size 的参数，表示一次输入多少个数据。 在 LSTM 模型中，输入数据必须是一批数据，为了区分LSTM中的批量数据和dataloader中的批量数据是否相同意义，LSTM 模型就通过这个参数的设定来区分。如果是相同意义的，就设置为True，如果不同意义的，设置为False。 torch.LSTM 中 batch_size 维度默认是放在第二维度，故此参数设置可以将 batch_size 放在第一维度。如：input 默认是(4,1,5)，中间的 1 是 batch_size，指定batch_first=True后就是(1,4,5)。所以，如果你的输入数据是二维数据的话，就应该将 batch_first 设置为True;
        self.encoder = nn.LSTM(self.n_features, self.hidden_size, batch_first=True,
                               num_layers=self.n_layers[0], bias=self.use_bias[0],
                               dropout=self.dropout[0])  # LSTM的堆叠层数为1
        self.encoder.to(var.device)
        self.decoder = nn.LSTM(self.n_features, self.hidden_size, batch_first=True,
                               num_layers=self.n_layers[1], bias=self.use_bias[1], dropout=self.dropout[1])
        self.decoder.to(var.device)
        self.hidden2output = nn.Linear(self.hidden_size, self.n_features)  # 输出线性层，用于预测评估
        self.hidden2output.to(var.device)

    def _init_hidden(self, batch_size):
        return (torch.Tensor(self.n_layers[0], batch_size, self.hidden_size).zero_().to(var.device),
                torch.Tensor(self.n_layers[0], batch_size, self.hidden_size).zero_().to(var.device)) # 长度为(1,20,5)的tensor用0填充

    def forward(self, ts_batch, return_latent: bool = False):  # ts_batch.size=(batch_size,sequence,X.shape[1])

        batch_size = ts_batch.shape[0]  # ts_batch.shape=[20,5,51]

        # 1. 对时间序列进行编码以利用最后一个隐藏状态。
        # 用零初始化,元组中有两个用零填充的大小为(1,20,5)的tensor
        (hidden_n, cell_n) = self._init_hidden(batch_size) 

        # 此处为.float()或.double()的模型.【output, (hn, cn) = rnn(input, (h0, c0))】,
        # 其中，output是最后一层lstm的每个词向量对应隐藏层的输出,其与层数无关，只与序列长度相关，
        # enc_hidden=(hn,cn)是所有层最后一个隐藏元和记忆元的输出
        _, (hidden_n, cell_n) = self.encoder(ts_batch.float().detach().requires_grad_(True),
                                     (hidden_n, cell_n))  

        # 2. 使用隐藏状态作为解码器LSTM的初始化
        dec_hidden = (hidden_n, cell_n)  # 是一个包含两个tensor的元组，tensor.size=(LSTM堆叠的层数=1,batch_size=20,hidden_size=5)

        # 3. 此外，使用此隐藏状态获取第一个输出，也就是重建的时间序列的最后一个点
        # 4. 反向重建时间序列
        #    * 使用真实数据训练解码器
        #    * 使用hidden2output进行预测
        output = torch.Tensor(ts_batch.size()).zero_().to(var.device)  # 设置输出值的维度与输入(20,5,51)相等，并用零填充
        for i in reversed(range(ts_batch.shape[1])):  # reversed 函数返回一个反转的迭代器,一共五条数据，一条一条的反向执行
            output[:, i, :] = self.hidden2output(
                dec_hidden[0][0, :])  # 预测。取大小为[1,20,5]的dec_hidden中第一个的全部数据[20,5],使用hidden2output转换为[20,51]

            if self.training:  # 使用训练数据训练解码器
                _, dec_hidden = self.decoder(ts_batch[:, i].unsqueeze(1).float(),
                                             dec_hidden)  # 训练。unsqueeze用于增加一个维度,将ts_batch转换为[20,1,51]
            else:  # 使用测试集预测模型时使用，对输出进行解码，取出预测出的第i条数据，解码
                _, dec_hidden = self.decoder(output[:, i].unsqueeze(1), dec_hidden)

        return ((hidden_n, cell_n), output) if return_latent else output
