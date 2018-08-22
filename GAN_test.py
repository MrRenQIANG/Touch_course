# _*_ coding:utf-8 _*_

import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt


torch.manual_seed(1)

BATCH_SIZE = 64
LR_G = 0.0001
LR_D = 0.0001
N_IDEAS = 5
ART_COMPONENTS = 15
PAINT_POINTS = np.vstack([np.linspace(-1, 1, ART_COMPONENTS) for _ in range(BATCH_SIZE)])  # 64*15

# show our beautiful painting range
# plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
# plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
# plt.legend(loc='upper right')
# plt.show()

def artist_works():  # painting from the famous artist (real target)一批著名画家的画
    a = np.random.uniform(1, 2, size=BATCH_SIZE)[:, np.newaxis]   # 从一个均匀分布[low,high)中随机采样
    paintings = a * np.power(PAINT_POINTS, 2) + a- 1
    # paintings = np.random.rand(BATCH_SIZE,ART_COMPONENTS)

    paintings = torch.from_numpy(paintings).float()
   #plt.plot(a);plt.show()
    return paintings             # 64*15

G = nn.Sequential(
    nn.Linear(N_IDEAS, 128),
    nn.ReLU(),
    nn.Linear(128, ART_COMPONENTS)          # 创造一幅画，维度是ART_COMPONENTS
)

D = nn.Sequential(
    nn.Linear(ART_COMPONENTS, 128),
    nn.ReLU(),
    nn.Linear(128,1),
    nn.Sigmoid(),                           # 判别，输出是一个数(概率）
)

OPT_G = torch.optim.Adam(G.parameters(), lr=LR_G)
OPT_D = torch.optim.Adam(D.parameters(), lr=LR_D)

plt.ion()

D_losses = []
G_losses = []

for step in range(10000):
    artist_painting = artist_works()
    G_ideas = torch.randn(BATCH_SIZE,N_IDEAS)
    G_painting = G(G_ideas)  # 64*15

    prob_artist0 = D(artist_painting)  # maxmize    size 64*1
    prob_artist1 = D(G_painting)  # minimize     size 64*1

    # tensorflow 或者torch中只能最小化误差，所以这里添加了负号
    D_loss = -torch.mean(torch.log(prob_artist0) + torch.log(1. - prob_artist1))
    G_loss = torch.mean(torch.log(1. - prob_artist1))

    D_losses.append(D_loss)
    G_losses.append(G_loss)
    OPT_D.zero_grad()
    D_loss.backward(retain_graph=True)
    OPT_D.step()

    OPT_G.zero_grad()
    G_loss.backward()
    OPT_G.step()



    if step % 100 ==0:
        plt.subplot(122)
        plt.cla()
        plt.plot(PAINT_POINTS[0], G_painting.data.numpy()[0], c ='#4AD631', lw=3, label='Generated painting')
        plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
        plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')

        plt.text(-.5, 2.3, 'D accuracy=%.2f (0.5 for D to converge)' % prob_artist0.data.numpy().mean(), fontdict={'size': 13})
        plt.text(-.5, 2, 'G accuracy=%.2f (0.5 for D to converge' % -D_loss.data.numpy(), fontdict={'size': 13})
        plt.ylim((0, 3));
        plt.legend(loc='upper right', fontsize=10);
    #     plt.subplot(121)
    #     plt.plot(D_losses, 'r-');
    #     plt.plot(G_losses, 'b-')
        plt.draw();
        plt.pause(0.01)

plt.ioff()
plt.show()