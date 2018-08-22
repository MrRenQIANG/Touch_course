# _*_ coding:utf-8 _*_

import torch
import torch.nn as nn
import torchvision
import torch.utils.data as Data
import matplotlib.pyplot as plt


torch.manual_seed(1)

EPOCH = 2
BATCH_SIZE = 64
TIME_STEP = 28
INPUT_SIZE = 28
LR = 0.001
DOWNLOAD_MNIST = False


# 包含traindata和trianlabel两个数据
train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,                                     # this is training data
    transform=torchvision.transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to
                                                    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=DOWNLOAD_MNIST,
)


# 包含testdata和testlabel两个数据
test_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST,
)

# 批训练
train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)


test_x = test_data.test_data.type(torch.FloatTensor)[:2000]/255
print(train_data.train_data.size())
test_y = test_data.test_labels[:2000]


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(         # if use nn.RNN(), it hardly learns
            input_size=INPUT_SIZE,
            hidden_size=64,         # rnn hidden unit
            num_layers=1,           # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.out = nn.Linear(64, 10)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state

        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        return out

rnn = RNN()
print(rnn)
# test_yy = rnn(test_x[:10].view(-1, 28, 28))
# print(torch.max(test_yy,1))       # torch.max)(a,0) 返回每一列中最大值的那个元素，且返回索引（返回最大元素在这一列的行索引）

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_func = torch.nn.CrossEntropyLoss()

losses = []
for epoch in range(EPOCH):
    for step, (x, b_y) in enumerate(train_loader):
        b_x = x.view(-1, 28, 28)
        output = rnn(b_x)

        loss = loss_func(output,b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.data.numpy())


        if step % 50 == 0:
            test_output = rnn(test_x)                   # (samples, time_step, input_size)
            pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
            accuracy = float((pred_y == test_y.data.numpy()).sum()) / float(test_y.size(0))
            print('Epoch', epoch, 'step', step, '*********** Train_Loss:%.4f' % loss.data.numpy(),
                  '*********Test_accuracy:.%4f' % accuracy)


test_yy = rnn(test_x[:10].view(-1, 28, 28))
# print(torch.max(test_yy, 1))
opt_y = torch.max(test_yy, 1)[1].data.numpy().squeeze()
pred_y = test_y[:10]

print(opt_y)
print(pred_y.data.numpy())

for i, loss in enumerate(losses):
    plt.plot(losses, 'r-')
plt.xlabel('steps')
plt.ylabel('loss')
plt.title('The CNN classifier loss about MNIST database')
plt.show()
