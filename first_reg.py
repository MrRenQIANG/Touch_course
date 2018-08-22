import torch
import matplotlib.pyplot as plt
import torch.nn.functional as f
import os

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = x.pow(2) + 0.2*torch.rand(x.size())

# figure
# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()
plt.ion()
plt.show()


class Net(torch.nn.Module):
    # nn.Linear(in_feature, out_hidden),in_features: size of each input sample ;out_features: size of each output sample
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # Linear的返回值是他的参数parameters
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = f.relu(self.hidden(x))
        x = self.predict(x)
        return x


net = Net(n_feature=1, n_hidden=10, n_output=1)
print(net)

optimizer = torch.optim.SGD(net.parameters(), lr=0.05, momentum=0.9)  # 传入 net 的所有参数, 学习率
loss_func = torch.nn.MSELoss()  # 预测值和真实值的误差计算公式 (均方差)

for i in range(500):
    prediction = net(x)

    loss = loss_func(prediction, y )  # 喂给 net 训练数据 x, 输出预测值
    optimizer.zero_grad()  # Clears the gradients of all optimized清空上一步的残余更新参数值
    loss.backward()  # 误差反向传播, 计算参数更新值
    optimizer.step()  # step()返回值为loss, 将参数更新值施加到 net 的 parameters 上

    if i % 5==0:
        plt.cla()
        plt.scatter(x.data.numpy(),y.data.numpy())
        plt.plot(x.data.numpy(),prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=% .4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)
# os.system('pause')