# _*_ coding:utf-8 _*_

import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable

# Hyper Parameters
TIME_STEP = 10
INPUT_SIZE = 1
LR = 0.01

# show data
# steps = np.linspace(0, np.pi*2, 100, dtype=np.float32)
steps = torch.unsqueeze(torch.linspace(0, np.pi*2, 100),dim=1).data.numpy()
x_np = np.sin(steps)
y_np = np.cos(steps)

plt.plot(steps, x_np, 'r-', label='targets(cos)')
plt.plot(steps, y_np, 'b-', label='input(sin)')
plt.legend(loc='best')
plt.show()


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        # **input** of shape `(seq_len, batch, input_size)
        self.rnn = nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=32,     # rnn hidden unit
            num_layers=3,       # number of rnn layer
            batch_first=True,   # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.out = nn.Linear(32, 1)

    def forward(self, x, h_state):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, hidden_size)
        r_out, h_state = self.rnn(x, h_state)   # 所有t的输出

        # 把每一步t的输出（32*1） 单独取出来做前向传递
        outs = []    # save all predictions
        for time_step in range(r_out.size(1)):    # calculate output for each time step
            outs.append(self.out(r_out[:, time_step, :]))
            # print(outs[0].size())        # outs -> 10*1*1
        return torch.stack(outs, dim=1), h_state

        # instead, for simplicity, you can replace above codes by follows
        # r_out = r_out.view(-1, 32)
        # outs = self.out(r_out)
        # return outs, h_state


rnn = RNN()
print(rnn)


optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_func = torch.nn.MSELoss()

h_state = None

plt.figure(1, figsize=(12, 5))
plt.ion()

for step in range(500):
    start, end = step * np.pi, (step+1) * np.pi
    steps = np.linspace(start, end, TIME_STEP, dtype=np.float32)
    x_np = np.sin(steps)
    y_np = np.cos(steps)

    # x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis])
    # y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis])

    x = Variable(torch.from_numpy(x_np[np.newaxis, :, np.newaxis]))
    # print(x.size())  # torch.Size([1, 10, 1])
    y = Variable(torch.from_numpy(y_np[np.newaxis, :, np.newaxis]))

    prediction , h_state = rnn(x, h_state)
    h_state = h_state.data

    loss = loss_func(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step > 100:
        plt.plot(steps, y_np.flatten(),'r-')
        plt.plot(steps, prediction.data.numpy().flatten(),'b-')
        plt.draw()
        plt.pause(0.1)


plt.ioff()
plt.show()
