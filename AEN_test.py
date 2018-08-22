# _*_ coding :utf-8 _*_

import torch
import numpy as np
import torch.nn as nn
import torchvision
import torch.utils.data as Data
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

EPOCH=10
BATCH_SIZE = 50
LR = 0.001
N_TEST_IMG = 5
DOWNLOAD_MNIST = False

# 包含traindata和trianlabel两个数据
train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,                                     # this is training data
    transform=torchvision.transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to
                                                    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=DOWNLOAD_MNIST,
)


# 批训练
train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)

print(train_data.train_data.size())                 # (60000, 28, 28)
print(train_data.train_labels.size())               # (60000)
# plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
# plt.title('%i' % train_data.train_labels[0])
# plt.show()


class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super(AutoEncoder,self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28,128),
            nn.Tanh(),
            nn.Linear(128,64),
            nn.Tanh(),
            nn.Linear(64,12),
            nn.Tanh(),
            nn.Linear(12,3),
        )

        self.decoder = nn.Sequential(
            nn.Linear(3,12),
            nn.Tanh(),
            nn.Linear(12,64),
            nn.Tanh(),
            nn.Linear(64,128),
            nn.Tanh(),
            nn.Linear(128,28*28),
        )

    def forward(self, x):
        encoded = self.encoder(x)

        decoded = self.decoder(encoded)

        return  encoded, decoded


autoencoder = AutoEncoder()
print(autoencoder)

optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
loss_func = nn.MSELoss()

# initialize figure
f, a = plt.subplots(2, N_TEST_IMG, figsize=(5, 2))  # subplots返回的值的类型为元组，其中包含两个元素：第一个为一个画布，第二个是子图
plt.ion()  # continuously plot

# original data (first row) for viewing
view_data = train_data.train_data[:N_TEST_IMG].view(-1, 28*28).type(torch.FloatTensor)/255
for i in range(N_TEST_IMG):
    a[0][i].imshow(np.reshape(view_data.data.numpy()[i], (28, 28)), cmap='gray');
    a[0][i].set_xticks(());
    a[0][i].set_yticks(())
    # plt.draw();
    # plt.pause(0.05)

for epoch in range(EPOCH):
    for step, (x, label) in enumerate(train_loader):
        b_x = x.view(-1, 28*28)
        b_y = x.view(-1, 28*28)

        _, decoded = autoencoder(b_x)

        loss = loss_func(decoded, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print('EPOCH: ', epoch, 'Loss: %.4f'% loss.data.numpy())

            # plot decoding image(second row)
            _, decoded_data = autoencoder(view_data)
            for i in range(N_TEST_IMG):
                a[1][i].clear()
                a[1][i].imshow(np.reshape(decoded_data.data.numpy()[i], (28, 28)), cmap='gray')
                a[1][i].set_xticks(())
                a[1][i].set_yticks(())
            plt.draw()
            plt.pause(0.05)

plt.ioff()
plt.show()


#visualize in 3D plot

view_data = train_data.train_data[:200].view(-1,28*28).type(torch.FloatTensor)/255
encoded_data, _ = autoencoder(view_data)

fig = plt.figure(2)
ax = Axes3D(fig)
X, Y, Z = encoded_data.data[:, 0].numpy(),encoded_data.data[:, 1].numpy(),encoded_data.data[:, 2].numpy()
values = train_data.train_labels[:200].numpy()
for x, y, z, v in zip(X,Y,Z,values):
    c = cm.rainbow(int(255 * v/9));
    ax.text(x, y, z, v, backgroundcolor=c)
ax.set_xlim(X.min(), X.max(0));ax.set_ylim(Y.min(), Y.max());ax.set_zlim(Z.min(), Z.max())
plt.show()