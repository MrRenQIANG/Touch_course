# _*_ coding:utf-8 _*_

import torch
# from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as f

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2*torch.rand(x.size())  # noisy y data (tensor), shape=(100, 1)


def save():
    net1 = torch.nn.Sequential(
        torch.nn.Linear(1,10),
        torch.nn.ReLU(),
        torch.nn.Linear(10,1)
    )
    optimizer = torch.optim.SGD(net1.parameters(),lr=0.02,momentum=0.9)
    loss_func = torch.nn.MSELoss()

    for i in range(500):
        prediction = net1(x)

        loss = loss_func(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # plot result
    plt.figure(1, figsize=(10, 3))
    plt.subplot(131)
    plt.title('Net1')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
    # plt.show()
    # 2 ways to save the net
    torch.save(net1, './net1.pkl')  # save entire net
    torch.save(net1.state_dict(), './net1_params.pkl')  # save only the parameters



def restore():
    # restore entire net1 to net2
    net2 = torch.load('./net1.pkl')
    prediction = net2(x)


    plt.figure(1, figsize=(10, 3))
    plt.subplot(132)
    plt.title('net2')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
    # plt.show()

def restore_params():  # 更快
    net3 = torch.nn.Sequential(
        torch.nn.Linear(1,10),
        torch.nn.ReLU(),
        torch.nn.Linear(10,1)
    )
    net3.load_state_dict(torch.load('./net1_params.pkl'))
    prediction = net3(x)
    # plot result
    plt.subplot(133)
    plt.title('Net3')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
    plt.show()


save()
restore()
restore_params()