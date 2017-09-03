import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLBP(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, sparsity=0.9):
        super().__init__(in_channels, out_channels, kernel_size, padding=1, bias=False)
        weights = next(self.parameters())
        matrix_proba = torch.FloatTensor(weights.data.shape).fill_(0.5)
        binary_weights = torch.bernoulli(matrix_proba) * 2 - 1
        mask_inactive = torch.rand(matrix_proba.shape) > sparsity
        binary_weights.masked_fill_(mask_inactive, 0)
        weights.data = binary_weights
        weights.requires_grad = False


class BlockLBP(nn.Module):

    def __init__(self, numChannels, numWeights):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(numChannels)
        self.conv_lbp = ConvLBP(numChannels, numWeights, kernel_size=3)
        self.conv_1x1 = nn.Conv2d(numWeights, numChannels, kernel_size=1)

    def forward(self, x):
        residual = x
        x = self.batch_norm(x)
        x = F.relu(self.conv_lbp(x))
        x = self.conv_1x1(x)
        x += residual
        return x


class Lbcnn(nn.Module):
    def __init__(self, numChannels=128, numWeights=512, full=512, depth=1):
        super().__init__()

        self.preprocess_block = nn.Sequential(
            nn.Conv2d(3, numChannels, kernel_size=3, padding=1),
            nn.BatchNorm2d(numChannels),
            nn.ReLU(inplace=True)
        )

        chain = [BlockLBP(numChannels, numWeights) for i in range(depth)]
        self.chained_blocks = nn.Sequential(*chain)
        self.pool = nn.AvgPool2d(kernel_size=5, stride=5)

        self.__fc1_dimension_in = numChannels * 6 * 6
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(self.__fc1_dimension_in, full)
        self.fc2 = nn.Linear(full, 10)

    def forward(self, x):
        x = self.preprocess_block(x)
        x = self.chained_blocks(x)
        x = self.pool(x)
        x = x.view(-1, self.__fc1_dimension_in)
        x = self.fc1(self.dropout(x))
        x = F.relu(x)
        x = self.fc2(self.dropout(x))
        return x
