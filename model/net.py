import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """
    Pytorch定义网络的标准格式
    """
    def __init__(self):
        """
        定义的一个简单卷积神经网络
        BatchNorm用来标准化，可以使得训练更加稳定
        """
        super(Net, self).__init__()
        self.num_channels = 32
        
        self.conv1 = nn.Conv2d(3, self.num_channels, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self.num_channels)
        self.conv2 = nn.Conv2d(self.num_channels, self.num_channels*2, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(self.num_channels*2)
        self.conv3 = nn.Conv2d(self.num_channels*2, self.num_channels*4, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(self.num_channels*4)

        # 全连接层将卷积层的输出转换为最终输出
        self.fc1 = nn.Linear(8*8*self.num_channels*4, self.num_channels*4)
        self.fcbn1 = nn.BatchNorm1d(self.num_channels*4)
        self.fc2 = nn.Linear(self.num_channels*4, 6)       
        self.dropout_rate = 0.5

    def forward(self, s):
        """
        此功能定义了我们如何使用网络的组件对输入batch进行操作

        Args:
            s: (Variable) 包含一批图像，尺寸为batch_size x 3 x 64 x 64（64为图片长宽，3为通道数）

        Returns:
            out: (Variable) 一个有着与类别数相同维数的数组，其中包含该图像属于各个类别的概率

        Note: 提供每个步骤后的尺寸
        """
        #                                                  -> batch_size x 3 x 64 x 64
        # 应用conv，batchnorm，maxpool和relu 三遍
        s = self.bn1(self.conv1(s))                         # batch_size x num_channels x 64 x 64
        s = F.relu(F.max_pool2d(s, 2))                      # batch_size x num_channels x 32 x 32
        s = self.bn2(self.conv2(s))                         # batch_size x num_channels*2 x 32 x 32
        s = F.relu(F.max_pool2d(s, 2))                      # batch_size x num_channels*2 x 16 x 16
        s = self.bn3(self.conv3(s))                         # batch_size x num_channels*4 x 16 x 16
        s = F.relu(F.max_pool2d(s, 2))                      # batch_size x num_channels*4 x 8 x 8

        # flatten 每个图像的输出结果
        s = s.view(-1, 8*8*self.num_channels*4)             # batch_size x 8*8*num_channels*4

        # 应用两个有dropout的全连接层
        s = F.dropout(F.relu(self.fcbn1(self.fc1(s))), 
            p=self.dropout_rate, training=self.training)    # batch_size x self.num_channels*4
        s = self.fc2(s)                                     # batch_size x 6

        # 在每个图像的输出上应用softmax
        return F.log_softmax(s, dim=1)


def loss_fn(outputs, labels):
    """
    给定输出和标签，计算交叉熵损失

    Args:
        outputs: (Variable) batch计算输出
        labels: (Variable) batch的真实标签

    Returns:
        loss (Variable): batch上的交叉熵损失
    """
    num_examples = outputs.size()[0]
    return -torch.sum(outputs[range(num_examples), labels])/num_examples


def accuracy(outputs, labels):
    """
    给定所有图像的输出和标签，计算精度

    Args:
        outputs: (np.ndarray) batch的sofemax的输出
        labels: (np.ndarray) batch的真实标签

    Returns: (float) 精度
    """
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs==labels)/float(labels.size)


# 此词典中所需的所有指标-在训练和评估循环中使用
metrics = {
    'accuracy': accuracy,
    # 可以增加更多指标，如每个类别的acc
}
