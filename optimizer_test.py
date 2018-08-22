import torch
import matplotlib.pyplot as plt
import torch.utils.data as Data
import torch.nn.functional as F


torch.manual_seed(1)

LR = 0.01
BATCH_SIZE = 32
EPOCH = 12

# fake data
x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
y = x.pow(2) + 0.1*torch.normal(torch.zeros(x.size()))

# plt.scatter(x.data.numpy(),y.data.numpy())
# plt.show()


torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(dataset=torch_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=True,
                         num_workers=2
                         )

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(1, 20)
        self.predict = torch.nn.Linear(20, 1)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x


net_GSD = Net()
net_Momentum = Net()
net_RMSprop = Net()
net_Adam = Net()
nets = [net_GSD, net_Momentum, net_RMSprop, net_Adam]

opt_SGD = torch.optim.SGD(net_GSD.parameters(),lr=LR)
opt_Momentum = torch.optim.SGD(net_GSD.parameters(),lr=LR, momentum=0.8)
opt_RMSprop = torch.optim.RMSprop(net_RMSprop.parameters(),lr=LR, alpha=0.9)
opt_Adam = torch.optim.Adam(net_Adam.parameters(),lr=LR,betas=(0.9, 0.99))
optimizers = [opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam]

loss_func = torch.nn.MSELoss()
losses = [[], [], [], []]

if __name__ == '__main__':
    for epoch in range(EPOCH):
        for step,(b_x, b_y) in enumerate(loader):
            for net,opt,los in zip(nets,optimizers,losses):
                '''
                程序中没有losses.append，但是执行完成之后losses却被添加了值：
                a =[1,2,3]
                b=[[],[],[]]
                for i,j in zip(a,b):
                    j.append(1)   
                b
                Out[6]: [[1], [1], [1]]
                a
                Out[7]: [1, 2, 3]
                '''
                output = net(b_x)
                loss = loss_func(output, b_y)
                opt.zero_grad()
                loss.backward()
                opt.step()

                los.append(loss.data[0])

    labels = ['GSD', 'Momentum', 'RMSprop', 'Adam']
    for i, l_his in enumerate(losses):
        plt.plot(l_his,label=labels[i])

    plt.legend(loc='best')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.show()
