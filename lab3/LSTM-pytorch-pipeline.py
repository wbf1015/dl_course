from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import string
import torch
import torch.nn as nn
import random
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib


all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)


def findFiles(path): return glob.glob(path)

# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)

# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(train_category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor

def randomEvaluatingExample():
    category = randomChoice(all_categories)
    line = randomChoice(test_category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

# Build the category_lines dictionary, a list of names per language
train_category_lines = {}
test_category_lines = {}
all_categories = [] #只是为了记录有哪些种类

for filename in findFiles('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    random.shuffle(lines)
    train_category_lines[category] = lines[:int(len(lines)*0.7)]
    test_category_lines[category] = lines[int(len(lines)*0.7):]

n_categories = len(all_categories)



def train(category_tensor, line_tensor):
    hidden, c = lstm.initHiddenAndC()
    lstm.zero_grad()
    for i in range(line_tensor.size()[0]):
        # 返回output, hidden以及细胞状态c
        output, hidden, c = lstm(line_tensor[i], hidden, c)
    # print(output.squeeze(0).squeeze(0).shape,' ',category_tensor.shape)
    loss = criterion(output.squeeze(0).squeeze(0), category_tensor)
    loss.backward()

    for p in lstm.parameters():
        p.data.add_(-learning_rate, p.grad.data)
    return output, loss.item()

def evaluate():
    lstm.eval()
    correct = 0
    count = 0
    total_loss = 0
    for category in all_categories:
        for line in test_category_lines[category]:
            category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
            line_tensor = lineToTensor(line)
            hidden, c = lstm.initHiddenAndC()
            lstm.zero_grad()
            for i in range(line_tensor.size()[0]):
                # 返回output, hidden以及细胞状态c
                output, hidden, c = lstm(line_tensor[i], hidden, c)

            guess, guess_i = categoryFromOutput(output)
            count += 1
            loss = criterion(output.squeeze(0).squeeze(0), category_tensor)
            total_loss += loss.item()
            if guess == category:
                correct+= 1
    lstm.train()
    return correct/count,total_loss/count

# 使用手写的lstm完成名字分类任务
class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0, bidirectional=False):
        super(MyLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional

        # Initialize weights
        self.weight_ih_l = nn.Parameter(torch.Tensor(input_size, 4 * hidden_size))
        self.weight_hh_l = nn.Parameter(torch.Tensor(hidden_size, 4 * hidden_size))
        self.bias_ih_l = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.bias_hh_l = nn.Parameter(torch.Tensor(4 * hidden_size))
        self._initialize_weights()

        # If bidirectional, create a copy of parameters for reverse direction
        if bidirectional:
            self.weight_ih_r = nn.Parameter(torch.Tensor(input_size, 4 * hidden_size))
            self.weight_hh_r = nn.Parameter(torch.Tensor(hidden_size, 4 * hidden_size))
            self.bias_ih_r = nn.Parameter(torch.Tensor(4 * hidden_size))
            self.bias_hh_r = nn.Parameter(torch.Tensor(4 * hidden_size))
            self._initialize_weights()

        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, input, hx=None):
        # Initialize hidden states if not provided
        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            hx = (torch.zeros(self.num_layers * num_directions, input.size(0), self.hidden_size, device=input.device),
                  torch.zeros(self.num_layers * num_directions, input.size(0), self.hidden_size, device=input.device))

        # Transpose input if batch_first
        if self.batch_first:
            input = input.transpose(0, 1)

        # Initialize variables
        output = []
        hy = []
        cx, hx = hx

        # Compute the forward pass for each timestep
        for i in range(input.size(0)):
            x = input[i]
            gates = x @ self.weight_ih_l + self.bias_ih_l + hx[-1] @ self.weight_hh_l + self.bias_hh_l
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            outgate = torch.sigmoid(outgate)
            cy = (forgetgate * cx) + (ingate * cellgate)
            hy.append(outgate * torch.tanh(cy))
            cx = cy
            hx = torch.stack(hy, dim=0)

            # Apply dropout
            hx = self.dropout_layer(hx)

            output.append(hx[-1])

        # Concatenate output and return
        output = torch.stack(output, dim=0)
        if self.bidirectional:
            output_rev = self._forward_reverse(input, hx)
            output = torch.cat((output, output_rev), dim=-1)
        if self.batch_first:
            output = output.transpose(0, 1)
        return output, (hy[-1], cy)

    def _forward_reverse(self, input, hx):
        # Transpose input if batch_first
        if self.batch_first:
            input = input.transpose(0, 1)

        # Initialize variables
        output = []
        hy = []
        cx, hx = hx

        # Compute the reverse pass for each timestep
        for i in reversed(range(input.size(0))):
            x = input[i]
            gates = x @ self.weight_ih_r + self.bias_ih_r + hx[-1] @ self.weight_hh_r + self.bias_hh_r
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            outgate = torch.sigmoid(outgate)
            cy = (forgetgate * cx) + (ingate * cellgate)
            hy.append(outgate * torch.tanh(cy))
            cx = cy
            hx = torch.stack(hy, dim=0)

            # Apply dropout
            hx = self.dropout_layer(hx)

            output.append(hx[-1])

        # Concatenate output and return
        output = torch.stack(output[::-1], dim=0)
        return output

    def _initialize_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        """初始化函数的参数与传统RNN相同"""
        super(LSTM, self).__init__()
        # 将hidden_size与num_layers传入其中
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 实例化预定义的nn.LSTM
        self.lstm = MyLSTM(input_size, hidden_size)
        # 实例化nn.Linear, 这个线性层用于将nn.RNN的输出维度转化为指定的输出维度
        self.linear = nn.Linear(hidden_size, output_size)
        # 实例化nn中预定的Softmax层, 用于从输出层获得类别结果
        self.softmax = nn.LogSoftmax(dim=-1)


    def forward(self, input, hidden, c):
        """在主要逻辑函数中多出一个参数c, 也就是LSTM中的细胞状态张量"""
        # 使用unsqueeze(0)扩展一个维度
        input = input.unsqueeze(0)
        # 将input, hidden以及初始化的c传入lstm中
        rr, (hn, c) = self.lstm(input, (hidden, c))
        # 最后返回处理后的rr, hn, c
        return self.softmax(self.linear(rr)), hn, c

    def initHiddenAndC(self):  
        """初始化函数不仅初始化hidden还要初始化细胞状态c, 它们形状相同"""
        c = hidden = torch.zeros(self.num_layers, 1, self.hidden_size)
        return hidden, c


learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn
n_hidden = 128
lstm = LSTM(n_letters, n_hidden, n_categories)
criterion = nn.NLLLoss()

n_iters = 1000000
print_every = 5000
plot_every = 1000
# Keep track of losses for plotting
current_loss = 0
all_losses = []
all_accuracy = []
start = time.time()
for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    # Print iter number, loss, name and guess
    if iter % print_every == 0:
        # guess, guess_i = categoryFromOutput(output)
        # correct = '✓' if guess == category else '✗ (%s)' % category
        # print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))
        acc,test_loss = evaluate()
        print('processing:',iter/n_iters,' acc on test dataset is:',acc,' loss on test dataset is:',loss)
        all_losses.append(test_loss)
        all_accuracy.append(acc)

plt.figure()
plt.plot(all_losses)
plt.savefig('LSTM-pytorch-loss.png')

plt.figure()
plt.plot(all_accuracy)
plt.savefig('LSTM-pytoch-acc.png')

