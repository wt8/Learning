import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.autograd import Variable

with open('data.txt', 'r') as f:
    data = f.read().split('\n')
    data = [row.split(',') for row in data][:-1]
    label0 = np.array([(float(row[0]), float(row[1])) for row in data if row[2] == '0'])  # 每个row为array的一行 size为n*2
    label1 = np.array([(float(row[0]), float(row[1])) for row in data if row[2] == '1'])
x0, y0 = label0[:, 0], label0[:, 1]
x1, y1 = label1[:, 0], label1[:, 1]
plt.plot(x0, y0, 'ro', label='label_0')
plt.plot(x1, y1, 'bo', label='label_1')
plt.legend(loc='best')

x = np.concatenate((label0, label1), axis=0)  # 按行拼接
x_data = torch.from_numpy(x).float()
y = [[0] for i in range(label0.shape[0])]
y += [[1] for i in range(label1.shape[0])]
y_data = torch.FloatTensor(y)


class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.lr = nn.Linear(2, 1)
        self.sm = nn.Sigmoid()

    def forward(self, x):
        x = self.lr(x)
        x = self.sm(x)
        return x


logistic_model = LogisticRegression()
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(logistic_model.parameters(), lr=1e-3, momentum=0.9)
for epoch in range(50000):
    x = x_data
    y = y_data
    out = logistic_model(x_data)
    loss = criterion(out, y_data)
    print_loss = loss
    mask = out.ge(0.5).float()
    correct = (mask == y).sum()
    acc = correct.item() / x.size(0)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch+1) % 1000 == 0:
        print('*'*10)
        print('epoch{}'.format(epoch+1))
        print('loss is {:.4f}'.format(print_loss))
        print('acc is {:.4f}'.format(acc))

w0,w1 = logistic_model.lr.weight[0]
w0 = w0.item()
w1 = w1.item()
b = logistic_model.lr.bias.item()
plot_x = np.arange(30,100,0.1)
plot_y = (-w0 * plot_x - b) / w1
plt.plot(plot_x, plot_y)
plt.show()


# w0, w1 = w, weight[1]
# b = logistic_model.line.bias.data[0]
#
# plt.plot(x0, y0, 'ro', label = 'label_0')
# plt.plot(x1, y1, 'bo', label = 'label_1')
# plt.legend(loc = 'best')
# plot_x = np.arange(30, 100, 0.1)
# plot_y = (-w0 * plot_x - b) / w1
# plt.plot(plot_x, plot_y)
# plt.show()