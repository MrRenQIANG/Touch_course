# _*_ coding:utf-8 _*_

import torch
import matplotlib.pyplot as plt
import torch.nn.functional as f

# 假数据
n_data = torch.ones(100, 2)         # 数据的基本形态
x0 = torch.normal(2*n_data, 1)      # 类型0 x data (tensor), shape=(100, 2)
y0 = torch.zeros(100)               # 类型0 y data (tensor), shape=(100, 1)
x1 = torch.normal(-2*n_data, 1)     # 类型1 x data (tensor), shape=(100, 1)
y1 = torch.ones(100)                # 类型1 y data (tensor), shape=(100, 1)

# 注意 x, y 数据的数据形式是一定要像下面一样 (torch.cat 是在合并数据)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # FloatTensor = 32-bit floating
y = torch.cat((y0, y1), 0).type(torch.LongTensor)    # LongTensor = 64-bit integer

# plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
# plt.show()


# class Net(torch.nn.Module):
#     def __init__(self,n_feature, n_hidden, n_out):
#         super(Net, self).__init__()
#         self.hidden = torch.nn.Linear(n_feature, n_hidden)
#         self.out = torch.nn.Linear(n_hidden, n_out)
#
#     def forward(self, x):
#         x =f.relu(self.hidden(x))
#         x =self.out(x)
#         return x

net = torch.nn.Sequential(
    torch.nn.Linear(2, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 2)
)

# net = Net(n_feature=2, n_hidden=10 , n_out=2)
print(net)

optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
loss_func = torch.nn.CrossEntropyLoss()

for i in range(100):
    out = net(x)

    loss = loss_func(out, y)
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()

    if i % 2 ==0:
        plt.cla()
        prediction = torch.max(f.softmax(out),1)[1]
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = sum(pred_y==target_y)/len(pred_y)
        plt.text(1.5,-4,'Accuracy=%.2f' % accuracy, fontdict={'size':20, 'color':'red'})
        plt.pause(0.5)


torch.save(net, './net.pkl')  # save entire net
plt.ioff()
plt.show()


'''
# 读取模型
net2 = torch.load('./net.pkl')
for i in range(100):
    out = net2(x)
    loss = loss_func(out, y)
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()

    if i % 2 ==0:
        plt.cla()
        prediction = torch.max(f.softmax(out),1)[1]
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = sum(pred_y==target_y)/len(pred_y)
        plt.text(1.5,-4,'Accuracy=%.2f' % accuracy, fontdict={'size':20, 'color':'red'})
        plt.pause(0.5)
        
plt.ioff()
plt.show()
'''