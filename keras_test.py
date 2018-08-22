# _*_ coding :utf-8 _*_

import torch
import tensorflow as tf
import  matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn as nn
import torchvision
import torch.utils.data as Data
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPool2D
# (x_train,y_train), (x_test,y_test) = tf.keras.datasets.mnist.load_data()

EPOCH = 10
# BATCH_SIZE = 50
LR = 0.001
N_TEST_IMG = 5
DOWNLOAD_MNIST = False
input_shape = (28, 28, 1)

# 包含traindata和trianlabel两个数据
train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,                                     # this is training data
    transform=torchvision.transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to
                                                    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=DOWNLOAD_MNIST,
)

test_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=False,                                     # this is test data
    transform=torchvision.transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to
                                                    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=DOWNLOAD_MNIST,
)

# 批训练
# train_loader = Data.DataLoader(
#     dataset=train_data,
#     batch_size=BATCH_SIZE,
#     shuffle=True
# )

print(train_data.train_data.size())                 # (60000, 28, 28)
print(train_data.train_labels.size())               # (60000)
# plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
# plt.title('%i' % train_data.train_labels[0])
# plt.show()
x_train_data, x_train_labels=train_data.train_data, train_data.train_labels
x_test_data, x_test_labels=test_data.test_data, test_data.test_labels

# 数据集在touch中已经做了数据处理，转换成0-1的区间
# x_train_data /= 255
# x_test_data /= 255

x_train_data = x_train_data.data.numpy().reshape(x_train_data.size()[0], 28, 28, 1).astype('float32')
x_test_data = x_test_data.data.numpy().reshape(x_test_data.size()[0], 28, 28, 1).astype('float32')
# print(x_train_data.size())                 # (60000, 28, 28)
# print(x_test_data.size())

x_train_labels = x_train_labels.data.numpy()
x_test_labels = x_test_labels.data.numpy()

model = Sequential()
model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x=x_train_data, y=x_train_labels, epochs=10, batch_size=256)
print(model.evaluate(x_test_data, x_test_labels, batch_size=256))
