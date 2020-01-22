import torch
from torch.autograd import Variable
import torch.nn as nn
import matplotlib.pyplot as plt


def make_features(x):
    # 将行转换成列
    x = x.unsqueeze(1)
    # cat按列拼接，每一行对那个x的一次、二次、三次
    return torch.cat([x ** i for i in range(1, 4)], 1)


# 将w行转换成列向量,完成矩阵相乘
W_target = torch.FloatTensor([0.5, 3, 2.4]).unsqueeze(1)
b_target = torch.FloatTensor([0.9])


def f(x):
    # 矩阵相乘
    return x.mm(W_target) + b_target[0]


def get_batch(batch_size=32):
    random = torch.randn(batch_size)
    x = make_features(random)
    y = f(x)
    return Variable(x), Variable(y)


class poly_model(nn.Module):
    def __init__(self):
        super(poly_model, self).__init__()
        self.poly = nn.Linear(3, 1)

    def forward(self, x):
        out = self.poly(x)
        return out


model = poly_model()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
epoch = 0
while True:
    batch_x, batch_y = get_batch()
    output = model(batch_x)
    loss = criterion(output, batch_y)
    print_loss = loss.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    epoch += 1
    if print_loss < 1e-3:
        break
model.eval()
print("Epoch = {}".format(epoch))

batch_x, batch_y = get_batch()
predict = model(batch_x)
a = predict - batch_y
y = torch.sum(a)
print('y = ', y)
predict = predict.data.numpy()
plt.plot(batch_x.numpy(), batch_y.numpy(), 'ro', label='Original data')
plt.plot(batch_x.numpy(), predict, 'b', ls='-', label='Fitting line')
plt.show()
