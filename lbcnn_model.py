import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

from torch.autograd import Variable
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
        x = self.batch_norm(x)
        residual = x
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

        chain = []
        for block_id in range(depth):
            basic_block = nn.Sequential(
                nn.BatchNorm2d(numChannels),
                ConvLBP(numChannels, numWeights, kernel_size=3),
                nn.ReLU(),
                nn.Conv2d(numWeights, numChannels, kernel_size=1)
            )
            chain.append(basic_block)
        self.chained_blocks = nn.Sequential(*chain)

        # self.chained_blocks = nn.Sequential(
        #     nn.BatchNorm2d(numChannels),
        #     ConvLBP(numChannels, numWeights, kernel_size=3),
        #     nn.ReLU(),
        #     nn.Conv2d(numWeights, numChannels, kernel_size=1)
        # )

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


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

BATCH_SIZE = 8


def do_train():
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=2)

    net = Lbcnn()
    net.cuda()
    # net = torch.load('lbcnn.pt')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(filter(lambda param: param.requires_grad, net.parameters()), lr=0.001, momentum=0.9)

    for epoch in range(5):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            inputs, labels = Variable(inputs), Variable(labels)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]
            if i % 200 == 0:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    torch.save(net, 'lbcnn.pt')
    print('Finished Training')


def do_test():
    net = torch.load('lbcnn.pt')
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                             shuffle=False, num_workers=2)
    correct = 0
    total = 0
    for data in testloader:
        images, labels = data
        images = images.cuda()
        labels = labels.cuda()
        outputs = net(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    for data in testloader:
        images, labels = data
        images = images.cuda()
        labels = labels.cuda()

        outputs = net(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i]
            class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))



if __name__ == '__main__':
    do_train()
    do_test()
